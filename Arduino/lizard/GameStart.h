#ifndef GameStart_h
#define GameStart_h

#include "Arduino.h"
#include "GamePhase.h"
#include "Relay.h"
#include "Player.h"

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
        Relay relay;
    private:
        int _switchRelayTime;
        Player _p1;
        Player _p2;
        LED _l1;
        LED _l2;
        LED _l3;
        LED _currentLED;
        void shuffle();
};

#endif
