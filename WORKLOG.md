# WORKLOG

Concise event wise / epoch wise logs of activities, marked by epoch timestamp ( `date +%s` )


## 2026-03-01
- [epoch:1772342379] Revist by user, and some cleanup, verbose logs marked archive and moved to `docs/`
- [epoch:1772344130] Pi5 tmux run: added interposer payload dump mode (`LIBUSB_TRACE_DUMP_DIR/LENS`), captured full good/replay `2608`+`9872` submit payloads; byte deltas are sparse and structured (2608: 6 bytes at `74-76,88-90`; 9872: 13 bytes at `232-234,249-252,7672-7674,7690-7692`), with replay side zeros at these fields.
