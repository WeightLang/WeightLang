use std::io::Write;
use std::os::unix::net::UnixStream;
use wgtime::{Request,receive_request};
#[test]
fn fixed_frames_preserve_boundaries_and_reject_truncation(){
    let(mut sender,mut receiver)=UnixStream::pair().unwrap();
    let mut frame=Vec::from(u32::MAX.to_le_bytes());for n in [7u64,8,9,10]{frame.extend_from_slice(&n.to_le_bytes());}
    sender.write_all(&frame).unwrap();wgtime::send(&mut sender,&Request::Stats).unwrap();
    assert!(matches!(receive_request(&mut receiver).unwrap(),Request::PredictBinary{handle:7,tensor:8,start:9,count:10}));
    assert!(matches!(receive_request(&mut receiver).unwrap(),Request::Stats));
    let(mut sender,mut receiver)=UnixStream::pair().unwrap();sender.write_all(&frame[..20]).unwrap();drop(sender);assert!(receive_request(&mut receiver).is_err());
    let(mut sender,mut receiver)=UnixStream::pair().unwrap();sender.write_all(&((wgtime::MAX_FRAME+1)as u32).to_le_bytes()).unwrap();assert!(receive_request(&mut receiver).is_err());
}
