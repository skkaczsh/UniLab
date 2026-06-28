from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Print pygame joystick axes for Xbox mapping.")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    import pygame

    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count == 0:
        raise SystemExit("No joystick detected by pygame.")
    if not 0 <= args.joystick_index < count:
        raise SystemExit(f"joystick-index {args.joystick_index} out of range; detected {count}.")

    joystick = pygame.joystick.Joystick(args.joystick_index)
    joystick.init()
    period = 1.0 / max(args.hz, 1.0)
    print(f"Joystick: {joystick.get_name()}")
    print("Move left stick and right stick. Press Ctrl-C to stop.")

    try:
        while True:
            pygame.event.pump()
            axes = [joystick.get_axis(axis_id) for axis_id in range(joystick.get_numaxes())]
            buttons = [
                button_id
                for button_id in range(joystick.get_numbuttons())
                if joystick.get_button(button_id)
            ]
            axis_text = " ".join(
                f"{axis_id}:{value:+.3f}" for axis_id, value in enumerate(axes)
            )
            button_text = ",".join(str(button_id) for button_id in buttons) or "-"
            print(f"\r axes [{axis_text}] buttons [{button_text}]      ", end="", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
