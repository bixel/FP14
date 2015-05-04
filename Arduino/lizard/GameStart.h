#ifndef GameStart_h
#define GameStart_h

#include "Arduino.h"
#include "GamePhase.h"

class GameStart : public GamePhase {
    public:
        GameStart(Relay& relay,
                  Player& player1,
                  Player& player2,
                  LED& l1,
                  LED& l2,
                  LED& l3,
                  int switchRelayTime = 600);
        void step();
    private:
        int _switchRelayTime;
        void shuffle();
};

#endif

