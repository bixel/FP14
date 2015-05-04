#include "GamePhase.h"

GamePhase::GamePhase(Relay& relay,
                     Player& p1,
                     Player& p2,
                     LED& l1,
                     LED& l2,
                     LED& l3) :
    _timer(0),
    _running(false),
    _relay(relay),
    _p1(p1),
    _p2(p2),
    _l1(l1),
    _l2(l2),
    _l3(l3),
    _currentLED(l1)
{}

bool GamePhase::isRunning() {
    return _running;
}

void GamePhase::run() {
    _running = true;
}

