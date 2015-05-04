#ifndef GameDisplay_h
#define GameDisplay_h

#include "Arduino.h"
#include "GamePhase.h"

class GameDisplay : public GamePhase {
    public:
        GameDisplay(Relay* relay,
                    Player* player1,
                    Player* player2,
                    LED* l1,
                    LED* l2,
                    LED* l3,
                    LED* l4,
                    int displayTime = 4000);
        void step();
        void run();
    private:
        int _displayTime;
        bool _draw;
        Player* _winner;
        void reset();
        void showPlayerSelection();
        void flash(LED* led);
};

#endif
