//! Cross-process recording inhibition. The kernel releases it on exit/crash.
use std::{
    fs::{self, File, OpenOptions},
    io,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::{Path, PathBuf},
};

pub fn lock_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("whisper-typer/recording.lock")
}

pub struct RecordingGuard {
    _file: File,
}
impl RecordingGuard {
    pub fn acquire() -> io::Result<Self> {
        Self::at_path(&lock_path())
    }
    pub fn at_path(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(path)?;
        // SAFETY: the descriptor is valid and owned throughout this guard.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { _file: file })
    }
}

pub fn active() -> bool {
    active_at(&lock_path())
}
pub fn active_at(path: &Path) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    // SAFETY: a shared nonblocking probe never alters the recorder's exclusive lock.
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
    result != 0 && io::Error::last_os_error().kind() == io::ErrorKind::WouldBlock
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader},
        process::{Command, Stdio},
    };
    fn test_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "whisper-recording-{label}-{}-{}.lock",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
    #[test]
    fn released_and_stale_files_do_not_mute() {
        let path = test_path("release");
        assert!(!active_at(&path));
        let guard = RecordingGuard::at_path(&path).unwrap();
        assert!(active_at(&path));
        assert!(RecordingGuard::at_path(&path).is_err());
        drop(guard);
        assert!(!active_at(&path));
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn crashed_recorder_automatically_releases_inhibition() {
        let path = test_path("crash");
        let mut child=Command::new("/usr/bin/python3").arg("-c").arg("import fcntl,sys; f=open(sys.argv[1],'w'); fcntl.flock(f,fcntl.LOCK_EX); print('ready',flush=True); sys.stdin.read()")
            .arg(&path).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        assert_eq!(line.trim(), "ready");
        assert!(active_at(&path));
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(!active_at(&path));
        fs::remove_file(path).unwrap();
    }
}
