const int relayPin = 2;
const int relayTime = 50;
int relayCurrentTime = 0;

const int button1Pin = 12;
const int button2Pin = 11;

const int LED1Pin = 13;

boolean paused = false;

void setup() {
  Serial.begin(9600);
  
  pinMode(relayPin, OUTPUT);
  pinMode(button1Pin, INPUT);
  pinMode(button2Pin, INPUT);
  pinMode(LED1Pin, OUTPUT);
}

void loop() {
  playing();
  shuffle();
  delay(10);
}

boolean playing() {
  int button1State;
  button1State = digitalRead(button1Pin);
  if (button1State == LOW) {
    if (relayCurrentTime == 0) {
      digitalWrite(relayPin, HIGH);
    } else if (relayCurrentTime == relayTime) {
      digitalWrite(relayPin, LOW);
    }
    relayCurrentTime++;
    if (relayCurrentTime >= 2*relayTime) {
      relayCurrentTime = 0;
    }
    return true;
  } else {
    return false;
  }
}

void shuffle() {
  int button2State;
  button2State = digitalRead(button2Pin);
  if(button2State == LOW) {
    digitalWrite(LED1Pin, HIGH);
  } else {
    digitalWrite(LED1Pin, LOW);
  }
}
