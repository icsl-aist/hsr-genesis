#!/usr/bin/env bash
set -e

# Start a virtual display if no real display is available and we're not
# forwarding X11 from the host.  This is required for pyglet (Genesis
# viewer import) even in headless mode.
if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=":99"
    Xvfb :99 -screen 0 1024x768x24 &
    XVFB_PID=$!
    # Wait for Xvfb to be ready
    for i in $(seq 10); do
        if xdpyinfo -display "$DISPLAY" &>/dev/null; then
            break
        fi
        sleep 0.2
    done
fi

exec "$@"
