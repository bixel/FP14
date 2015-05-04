#include "GameDefense.h"

GameDefense::GameDefense(Relay& relay,
                         Player& p1,
                         Player& p2,
                         LED& l1,
                         LED& l2,
                         LED& l3,
                         int defenseTime) :
        GamePhase(relay, p1, p2, l1, l2, l3),
        _defenseTime(defenseTime),
{}

