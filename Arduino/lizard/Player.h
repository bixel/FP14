#ifndef Player_h
#define Player_h

#include "Arduino.h"
#include "Button.h"
#include "LED.h"

class Player {
    public:
        Player(int color, const Button& button);
        Button button;
        LED selectedLED;
        int getColor();
    private:
        int _color;
        int _score;
};

#endif
