//! The body IMU on I²C, for a robot built without the `imu_to_dxl` board.
//!
//! The board is an LSM6DSV16X with a Dynamixel front end. This module is the same chip wired
//! straight to an I²C bus — a breakout on the HAT's Qwiic connector — producing the same
//! [`IMU_BLOCK_LEN`]-byte block the board serves at address 124, so everything downstream of the
//! bus read is untouched: [`crate::imu::SflpDecoder`], its mount, the median filter, the stale
//! tracker and `ready()` all see what they see on a robot with the board.
//!
//! **Why the block is free.** The chip's FIFO records are a tag byte and six data bytes. A gyro
//! record's six bytes are x/y/z as `i16` LE raw counts, and an SFLP game-rotation record's are
//! x/y/z as half floats — which is exactly the board's block, gyro then quaternion. Nothing is
//! converted, so nothing can be converted wrong.
//!
//! **Why a thread.** The control loop's read has a 20 ms budget and this bus is shared: on a
//! `zero3` the ToF sits on the same I²C3, and one 8×8 frame from it holds the bus for tens of
//! milliseconds. Reading the chip inline would hand the tick's timing to whatever `tofd` is
//! doing. Instead a thread drains the FIFO at [`POLL_HZ`] and keeps the newest block, and the
//! tick takes it without touching the bus. A tick that lands between two samples sees the same
//! block twice, which the stale tracker counts as the ordinary hiccup it also is on the board.
//!
//! The chip configuration is the one `robotd`'s head IMU reader uses (itself `imu_to_dxl`'s),
//! at 120 Hz rather than 60: gyro ±500 dps, accel ±4 g, SFLP game vector. Only gyro and SFLP are
//! batched — the block has no room for accel and the fusion does not need it batched.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::imu::IMU_BLOCK_LEN;
use crate::io::{IoError, Result};

/// Where the tick's IMU block comes from when it is not the Dynamixel bus.
///
/// A trait rather than the I²C reader by name so the bus can be tested against a fake, and so
/// the bus module does not grow a dependency on Linux.
pub trait ImuBlockSource: Send {
    /// The newest block. An error when there has been no fresh sample for longer than a tick
    /// can usefully coast — the same failure a silent board on the bus produces.
    fn latest(&mut self) -> Result<[u8; IMU_BLOCK_LEN]>;
    /// Is the chip there and producing? The I²C counterpart of pinging id 200.
    fn answers(&mut self) -> Result<bool>;
    /// Where it is, for the journal: `"/dev/i2c-3 0x6b"`.
    fn location(&self) -> String;
}

/// How often the reader drains the FIFO. Faster than [`SAMPLE_HZ`] so a sample is picked up
/// within a few milliseconds of being made, and so a FIFO with one record in it is the norm.
pub const POLL_HZ: u32 = 200;

/// The SFLP and gyro batch rate. The SFLP offers 15, 30, 60, 120, 240 and 480 Hz; 120 is the
/// first comfortably above the 50 Hz loop, so a tick's block is never more than ~8 ms old.
pub const SAMPLE_HZ: u32 = 120;

/// A block older than this is not a measurement any more: the reader has lost the chip.
/// Generous on purpose — a ToF frame holding the bus is tens of milliseconds, and that is a
/// hiccup the tick should coast over, not a fault.
pub const MAX_AGE: Duration = Duration::from_millis(250);

/// How long [`ImuBlockSource::answers`] waits for the first sample after the reader starts:
/// bring-up (reset, trimming reload, configure) is ~50 ms and the first SFLP output follows.
const FIRST_SAMPLE_WAIT: Duration = Duration::from_millis(600);

// ---- Registers (ST DS13510), as in robotd's head IMU reader ---------------------------------
const FUNC_CFG_ACCESS: u8 = 0x01;
const FIFO_CTRL3: u8 = 0x09;
const FIFO_CTRL4: u8 = 0x0a;
const WHO_AM_I: u8 = 0x0f;
const CTRL1: u8 = 0x10;
const CTRL2: u8 = 0x11;
const CTRL3: u8 = 0x12;
const CTRL6: u8 = 0x15;
const CTRL8: u8 = 0x17;
const CTRL9: u8 = 0x18;
const FIFO_STATUS1: u8 = 0x1b;
const FIFO_DATA_OUT_TAG: u8 = 0x78;
const EMB_FUNC_EN_A: u8 = 0x04;
const EMB_FUNC_FIFO_EN_A: u8 = 0x44;
const SFLP_ODR: u8 = 0x5e;
const EMB_FUNC_INIT_A: u8 = 0x66;
/// SFLP_ODR[5:3]: 011 is 120 Hz. The other bits are reserved and kept.
const SFLP_ODR_MASK: u8 = 0b0011_1000;
const SFLP_ODR_120HZ: u8 = 0b011 << 3;
const SFLP_GAME_BIT: u8 = 1 << 1;

const WHO_AM_I_VALUE: u8 = 0x70;
const CTRL3_BOOT: u8 = 1 << 7;
const CTRL3_SW_RESET: u8 = 1 << 0;

/// User-bank configuration, written in this order and read back.
const CONFIG: &[(u8, u8)] = &[
    (CTRL3, 0x44),      // BDU + IF_INC (burst auto-increment)
    (CTRL6, 0x02),      // gyro ±500 dps — the scale `SflpDecoder` decodes
    (CTRL8, 0x41),      // accel ±4 g, LPF2 ODR/20
    (CTRL9, 0x08),      // LPF2_XL_EN
    (CTRL1, 0x06),      // accel 120 Hz, high performance (SFLP needs the sensors >= its rate)
    (CTRL2, 0x06),      // gyro  120 Hz, high performance
    (FIFO_CTRL3, 0x60), // BDR_GY[7:4] = 120 Hz, BDR_XL[3:0] = not batched
    (FIFO_CTRL4, 0x06), // continuous: the oldest records go if the reader falls behind
];

const TAG_GYRO: u8 = 0x01;
const TAG_SFLP_GAME: u8 = 0x13;
const RECORD: usize = 7;
/// More than a poll ever finds (one or two records at 200 Hz), less than the FIFO's 512 words,
/// so a backlog after a stall drains in a few polls rather than one long transfer.
const MAX_WORDS: usize = 32;

/// Fold a burst of FIFO records into the newest block, if it holds a quaternion.
///
/// A quaternion is paired with the most recent gyro record before it, which may be one from an
/// earlier burst — hence `gyro` outliving a call. Before any gyro record at all the gyro half is
/// zero, which reads as a robot at rest; the decoder is not `ready()` that early anyway.
pub fn fold(burst: &[u8], gyro: &mut [u8; 6]) -> Option<[u8; IMU_BLOCK_LEN]> {
    let mut newest = None;
    let (records, _) = burst.as_chunks::<RECORD>();
    for record in records {
        match record[0] >> 3 {
            TAG_GYRO => gyro.copy_from_slice(&record[1..]),
            TAG_SFLP_GAME => {
                let mut block = [0u8; IMU_BLOCK_LEN];
                block[..6].copy_from_slice(gyro);
                block[6..].copy_from_slice(&record[1..]);
                newest = Some(block);
            }
            _ => {}
        }
    }
    newest
}

/// What the reader thread shares with the tick.
#[derive(Debug)]
struct Shared {
    block: Option<[u8; IMU_BLOCK_LEN]>,
    at: Option<Instant>,
    /// Why there is no chip, for the error a stale `latest` returns.
    status: String,
    stop: bool,
}

/// An LSM6DSV16X on I²C, read on a thread of its own.
///
/// Dropping it stops the thread: `robotd` opens the bus afresh on every startup attempt, and
/// each attempt must not leave a reader behind.
pub struct I2cBodyImu {
    shared: Arc<Mutex<Shared>>,
    location: String,
    started: Instant,
}

impl I2cBodyImu {
    /// Start reading `address` on `bus`. Never fails: a chip that is not there yet is reported
    /// by [`ImuBlockSource::answers`], and the thread keeps looking for it.
    pub fn start(bus: &str, address: u16) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            block: None,
            at: None,
            status: "starting".to_owned(),
            stop: false,
        }));
        let location = format!("{bus} {address:#04x}");
        let thread_shared = Arc::clone(&shared);
        let thread_bus = bus.to_owned();
        let spawned = std::thread::Builder::new()
            .name("body-imu".into())
            .spawn(move || reader::run(&thread_bus, address, &thread_shared));
        if let Err(e) = spawned {
            shared.lock().unwrap().status = format!("cannot start the reader thread: {e}");
        }
        Self {
            shared,
            location,
            started: Instant::now(),
        }
    }
}

impl Drop for I2cBodyImu {
    fn drop(&mut self) {
        if let Ok(mut s) = self.shared.lock() {
            s.stop = true;
        }
    }
}

impl ImuBlockSource for I2cBodyImu {
    fn latest(&mut self) -> Result<[u8; IMU_BLOCK_LEN]> {
        let s = self.shared.lock().unwrap();
        match (s.block, s.at) {
            (Some(block), Some(at)) if at.elapsed() <= MAX_AGE => Ok(block),
            (Some(_), Some(at)) => Err(IoError::Bus(format!(
                "body IMU at {}: no sample for {} ms ({})",
                self.location,
                at.elapsed().as_millis(),
                s.status
            ))),
            _ => Err(IoError::Bus(format!(
                "body IMU at {}: no sample yet ({})",
                self.location, s.status
            ))),
        }
    }

    fn answers(&mut self) -> Result<bool> {
        // The reader started with this source; give it the time a bring-up takes before
        // calling the chip absent, or every first attempt reports a missing IMU.
        loop {
            if self.latest().is_ok() {
                return Ok(true);
            }
            if self.started.elapsed() >= FIRST_SAMPLE_WAIT {
                return Ok(false);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn location(&self) -> String {
        self.location.clone()
    }
}

#[cfg(target_os = "linux")]
mod reader {
    use std::fs::{File, OpenOptions};
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;

    use super::*;

    /// `I2C_SLAVE` from linux/i2c-dev.h.
    const I2C_SLAVE: libc::c_ulong = 0x0703;
    const RETRY_MIN: Duration = Duration::from_millis(200);
    const RETRY_MAX: Duration = Duration::from_secs(5);

    struct Chip {
        file: File,
    }

    impl Chip {
        fn open(bus: &str, address: u16) -> std::io::Result<Self> {
            let file = OpenOptions::new().read(true).write(true).open(bus)?;
            // SAFETY: I2C_SLAVE takes the address by value; the fd is owned and open.
            if unsafe { libc::ioctl(file.as_raw_fd(), I2C_SLAVE, libc::c_ulong::from(address)) } < 0
            {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Self { file })
        }

        fn write(&mut self, reg: u8, value: u8) -> std::io::Result<()> {
            self.file.write_all(&[reg, value])
        }

        fn read(&mut self, reg: u8, buf: &mut [u8]) -> std::io::Result<()> {
            self.file.write_all(&[reg])?;
            self.file.read_exact(buf)
        }

        fn read1(&mut self, reg: u8) -> std::io::Result<u8> {
            let mut b = [0u8];
            self.read(reg, &mut b)?;
            Ok(b[0])
        }

        fn set_bits(&mut self, reg: u8, mask: u8) -> std::io::Result<()> {
            let v = self.read1(reg)?;
            self.write(reg, v | mask)
        }

        /// `imu_to_dxl`'s bring-up: reset, reload the trimming, configure, read back, start
        /// SFLP. Configuring before the trimming has loaded reads half the true gyro rate with
        /// a correct quaternion — the "drunk robot" of `imu_to_dxl` fw 6.
        fn bring_up(&mut self) -> std::io::Result<()> {
            let who = self.read1(WHO_AM_I)?;
            if who != WHO_AM_I_VALUE {
                return Err(std::io::Error::other(format!(
                    "WHO_AM_I {who:#04x}, not an LSM6DSV16X"
                )));
            }
            self.write(CTRL3, CTRL3_SW_RESET)?;
            for _ in 0..20 {
                std::thread::sleep(Duration::from_millis(1));
                if self.read1(CTRL3)? & CTRL3_SW_RESET == 0 {
                    break;
                }
            }
            self.write(CTRL3, CTRL3_BOOT | 0x44)?;
            std::thread::sleep(Duration::from_millis(30));
            for &(reg, value) in CONFIG {
                self.write(reg, value)?;
            }
            for &(reg, value) in CONFIG {
                let got = self.read1(reg)?;
                if got != value {
                    return Err(std::io::Error::other(format!(
                        "register {reg:#04x} reads {got:#04x}, wrote {value:#04x}"
                    )));
                }
            }
            self.write(FUNC_CFG_ACCESS, 0x80)?;
            let sflp = (|| {
                let odr = self.read1(SFLP_ODR)?;
                self.write(SFLP_ODR, (odr & !SFLP_ODR_MASK) | SFLP_ODR_120HZ)?;
                self.set_bits(EMB_FUNC_EN_A, SFLP_GAME_BIT)?;
                self.set_bits(EMB_FUNC_FIFO_EN_A, SFLP_GAME_BIT)?;
                self.set_bits(EMB_FUNC_INIT_A, SFLP_GAME_BIT)
            })();
            // Back to the user bank whatever happened, or every later read lands in the wrong one.
            self.write(FUNC_CFG_ACCESS, 0x00)?;
            sflp
        }
    }

    fn stopped(shared: &Mutex<Shared>) -> bool {
        shared.lock().map(|s| s.stop).unwrap_or(true)
    }

    fn set_status(shared: &Mutex<Shared>, status: String) {
        if let Ok(mut s) = shared.lock() {
            s.status = status;
        }
    }

    /// Sleep in short slices so a drop is honoured promptly.
    fn nap(dur: Duration, shared: &Mutex<Shared>) {
        let mut left = dur;
        while left > Duration::ZERO && !stopped(shared) {
            let step = left.min(Duration::from_millis(50));
            std::thread::sleep(step);
            left = left.saturating_sub(step);
        }
    }

    pub(super) fn run(bus: &str, address: u16, shared: &Mutex<Shared>) {
        let period = Duration::from_secs_f64(1.0 / f64::from(POLL_HZ));
        let mut backoff = RETRY_MIN;
        let mut announced = false;
        while !stopped(shared) {
            let mut chip = match Chip::open(bus, address).and_then(|mut c| c.bring_up().map(|()| c))
            {
                Ok(chip) => chip,
                Err(e) => {
                    set_status(shared, format!("no LSM6DSV16X: {e}"));
                    if !announced {
                        tracing::warn!(bus, address, error = %e, "body IMU not found on I2C; retrying");
                        announced = true;
                    }
                    nap(backoff, shared);
                    backoff = (backoff * 2).min(RETRY_MAX);
                    continue;
                }
            };
            tracing::info!(
                bus,
                address,
                hz = SAMPLE_HZ,
                "body IMU found: LSM6DSV16X on I2C"
            );
            set_status(shared, "reading".to_owned());
            backoff = RETRY_MIN;
            announced = false;

            let mut gyro = [0u8; 6];
            let mut burst = vec![0u8; MAX_WORDS * RECORD];
            let error = loop {
                if stopped(shared) {
                    return;
                }
                let tick = Instant::now();
                let mut level = [0u8; 2];
                if let Err(e) = chip.read(FIFO_STATUS1, &mut level) {
                    break e;
                }
                let words =
                    (usize::from(level[0]) | (usize::from(level[1] & 0x01) << 8)).min(MAX_WORDS);
                if words > 0 {
                    let bytes = &mut burst[..words * RECORD];
                    if let Err(e) = chip.read(FIFO_DATA_OUT_TAG, bytes) {
                        break e;
                    }
                    if let Some(block) = fold(bytes, &mut gyro)
                        && let Ok(mut s) = shared.lock()
                    {
                        s.block = Some(block);
                        s.at = Some(Instant::now());
                    }
                }
                let elapsed = tick.elapsed();
                if elapsed < period {
                    std::thread::sleep(period - elapsed);
                }
            };
            set_status(shared, format!("read failed: {error}"));
            tracing::warn!(bus, address, error = %error, "body IMU read failed; reopening");
            nap(backoff, shared);
            backoff = (backoff * 2).min(RETRY_MAX);
        }
    }
}

/// Off Linux there is no I²C bus: say so, once, and never produce a sample.
#[cfg(not(target_os = "linux"))]
mod reader {
    use super::*;

    pub(super) fn run(_bus: &str, _address: u16, shared: &Mutex<Shared>) {
        if let Ok(mut s) = shared.lock() {
            s.status = "I2C exists only on Linux".to_owned();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu::SflpDecoder;

    fn record(tag: u8, data: [u8; 6]) -> [u8; RECORD] {
        let mut r = [0u8; RECORD];
        r[0] = tag << 3;
        r[1..].copy_from_slice(&data);
        r
    }

    /// The block is the gyro record's bytes then the quaternion's, untouched.
    #[test]
    fn a_gyro_then_a_quaternion_make_one_block() {
        let mut burst = Vec::new();
        burst.extend(record(TAG_GYRO, [1, 2, 3, 4, 5, 6]));
        burst.extend(record(TAG_SFLP_GAME, [7, 8, 9, 10, 11, 12]));
        let mut gyro = [0u8; 6];
        assert_eq!(
            fold(&burst, &mut gyro),
            Some([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        );
    }

    /// Several samples in one burst: the newest wins, paired with the gyro just before it.
    #[test]
    fn the_newest_quaternion_wins() {
        let mut burst = Vec::new();
        burst.extend(record(TAG_GYRO, [1; 6]));
        burst.extend(record(TAG_SFLP_GAME, [2; 6]));
        burst.extend(record(TAG_GYRO, [3; 6]));
        burst.extend(record(TAG_SFLP_GAME, [4; 6]));
        let mut gyro = [0u8; 6];
        let block = fold(&burst, &mut gyro).unwrap();
        assert_eq!(&block[..6], &[3; 6]);
        assert_eq!(&block[6..], &[4; 6]);
    }

    /// A burst that opens on a quaternion pairs it with the gyro the previous burst left.
    #[test]
    fn a_quaternion_first_uses_the_last_gyro() {
        let mut gyro = [9u8; 6];
        let block = fold(&record(TAG_SFLP_GAME, [5; 6]), &mut gyro).unwrap();
        assert_eq!(&block[..6], &[9; 6]);
    }

    /// Gyro alone, accel, timestamps and a torn record make no block.
    #[test]
    fn no_quaternion_no_block() {
        let mut burst = record(TAG_GYRO, [1; 6]).to_vec();
        burst.extend(record(0x02, [2; 6]));
        burst.extend([TAG_SFLP_GAME << 3, 1, 2]);
        let mut gyro = [0u8; 6];
        assert_eq!(fold(&burst, &mut gyro), None);
        assert_eq!(gyro, [1; 6], "the gyro is still remembered");
    }

    /// End to end through the board's own decoder: the head IMU's recorded burst from a real
    /// chip (robotd `head_imu` tests, 2026-10-06), folded here, decodes to the same gyro rate.
    #[test]
    fn a_real_burst_decodes_through_the_board_decoder() {
        let mut burst = Vec::new();
        burst.extend(record(TAG_GYRO, [0xf1, 0x06, 0xb8, 0x0b, 0x1f, 0xee]));
        burst.extend(record(TAG_SFLP_GAME, [0xc9, 0x39, 0x7b, 0x18, 0x8a, 0x16]));
        let mut gyro = [0u8; 6];
        let block = fold(&burst, &mut gyro).unwrap();
        let mut d = SflpDecoder::new(SflpDecoder::BETA_MOUNT);
        d.decode(&block);
        d.decode(&block);
        let out = d.decode(&block);
        // 0x06f1 = 1777 LSB × 17.5 mdps = 0.5428 rad/s about the chip's x, which BETA_MOUNT
        // leaves as the trunk's x.
        assert!((out.gyro[0] - 0.5428).abs() < 1e-3, "{:?}", out.gyro);
        let [w, x, y, z] = out.quat;
        assert!(((w * w + x * x + y * y + z * z).sqrt() - 1.0).abs() < 1e-3);
    }

    /// A source with no chip behind it says so rather than inventing a block.
    #[test]
    fn a_source_with_nothing_behind_it_reports_no_sample() {
        let mut s = I2cBodyImu::start("/dev/this-bus-does-not-exist", 0x6b);
        assert!(s.latest().is_err());
        assert_eq!(s.location(), "/dev/this-bus-does-not-exist 0x6b");
    }
}
