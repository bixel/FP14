#include "GamePhase.h"

GamePhase::GamePhase() {
    _timer = 0;
    _running = false;
}

bool GamePhase::isRunning() {
    return _running;
}

void GamePhase::run() {
    _running = true;
}
