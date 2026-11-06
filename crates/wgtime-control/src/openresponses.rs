//! Native OpenResponses 2026-04-24 client and bounded harness state.
//! Protocol source: openresponses/openresponses@dce9ddb91921427c2ba06901c7dfaf0cc7056799.
//! Provider extensions are retained as data and never dispatched as tools.
#[path = "openresponses/budget.rs"]
pub mod budget;
#[path = "openresponses/profiles.rs"]
pub mod profiles;
#[path = "openresponses/session.rs"]
pub mod session;
#[path = "openresponses/sse.rs"]
pub mod sse;
#[path = "openresponses/transport.rs"]
pub mod transport;
#[path = "openresponses/wire.rs"]
pub mod wire;

pub const SPEC_VERSION: &str = "2026-04-24";
pub const SPEC_REVISION: &str = "dce9ddb91921427c2ba06901c7dfaf0cc7056799";
