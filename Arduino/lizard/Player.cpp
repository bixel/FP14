#include "Player.h"

Player::Player(int color, const Button& button) :
    button(button), selectedLED(NULL)
{
    _color = color;
    _score = 0;
}

int Player::getColor() {
    return _color;
}
