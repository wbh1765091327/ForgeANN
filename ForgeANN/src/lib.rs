// Allow for specialization
#![allow(incomplete_features)]
#![allow(clippy::module_inception)]
#![cfg_attr(test, allow(clippy::unused_io_amount))]
#![feature(stmt_expr_attributes)]
#![feature(specialization)]

pub mod utils;

pub mod forgeann;

pub mod model;

pub mod common;

pub mod index;
