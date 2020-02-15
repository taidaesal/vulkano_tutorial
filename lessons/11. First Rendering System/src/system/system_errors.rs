use std::fmt;

#[derive(Clone)]
pub enum SystemErrorType {
    ResizeFailure
}

#[derive(Clone)]
pub struct SystemError {
    err_type: SystemErrorType
}

impl SystemError {
    pub fn new(err_type: SystemErrorType) -> SystemError {
        SystemError{ err_type }
    }
}


impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_msg = match self.err_type {
            SystemErrorType::ResizeFailure => "Error occurred while resizing swapchain",
        };

        write!(f, "{}", err_msg)
    }
}

impl fmt::Debug for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_msg = match self.err_type {
            SystemErrorType::ResizeFailure => "Error occurred while resizing swapchain",
        };

        write!(f, "{}", err_msg)
    }
}