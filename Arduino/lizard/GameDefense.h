#ifndef GameDefense_h
#define GameDefense_h

#include "Arduino.h"
#include "GamePhase.h"

class GameDefense : public GamePhase {
    public:
        GameDefense(Relay* relay,
                    Player* player1,
                    Player* player2,
                    LED* l1,
                    LED* l2,
                    LED* l3,
                    LED* l4,
                    int defenseTime = 600);
        void step();
        void run(LED* currentLED, Player* currentPlayer);
        void reset();
    private:
        int _defenseTime;
        Player* _currentPlayer;
        void shuffle();
        void flashing();
};

#endif
