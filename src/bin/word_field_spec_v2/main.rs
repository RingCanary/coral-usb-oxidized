mod args;
mod fit;
mod model;
mod predict;
mod report;
mod util;

use args::parse_args;

fn main() {
    let config = match parse_args() {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("error: {}", err);
            std::process::exit(2);
        }
    };

    if let Err(err) = report::run(config) {
        eprintln!("error: {}", err);
        std::process::exit(1);
    }
}
