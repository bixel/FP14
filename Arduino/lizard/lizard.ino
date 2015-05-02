// Define counters for timing
const int relayPin = 2;
const int relayTime = 50;
int relayCurrentTime = 0;

const int shuffleTime = 10;
int shuffleCurrentTime = 0;

const int counterTime = 200;
const int counterTimeGain = 20;
int counterShuffleTime = counterTime;
int counterCurrentTime = 0;

// Set stages
const int stagePlay = 0; // This is the 1vs1 stage
const int stageCounter = 1; // This is the counter-measure stage
const int stageDisplay = 2; // Display game result here
int currentStage = stagePlay;

// Some Pin Definitions
const int button1Pin = 12;
const int button2Pin = 11;
const int LED1Pin = 13;
const int LED2Pin = 10;
const int LED3Pin = 9;
const int LEDStatusPin = 7;
const int LED1Prop = 300;
const int LED2Prop = 600;
const int LED3Prop = 900;
const int LEDStatusProp = 1000;

// some status variables
int currentPin = 0;
int currentPlayer = 0;
int selectedPin = currentPin;
boolean button1Pressed = false;
boolean button2Pressed = false;

void setup() {
  Serial.begin(9600);
  
  pinMode(relayPin, OUTPUT);
  pinMode(button1Pin, INPUT);
  pinMode(button2Pin, INPUT);
  pinMode(LED1Pin, OUTPUT);
  pinMode(LED2Pin, OUTPUT);
  pinMode(LED3Pin, OUTPUT);
  pinMode(LEDStatusPin, OUTPUT);
}

void loop() {
  // Get the button status
  button1Pressed = digitalRead(button1Pin) == LOW ? 1 : 0;
  button2Pressed = digitalRead(button2Pin) == LOW ? 1 : 0;
  
  if (currentStage == stagePlay) {
    if (button1Pressed || button2Pressed){
      selectedPin = currentPin;
      currentStage++;
    } else {
      currentPlayer = playing();
      shuffle(shuffleTime);
    }
  }
  
  if (currentStage == stageCounter) {
    counter();
  }
  
  if (currentStage == stageDisplay) {
  
  }
  
  // delay for smoother gameplay
  delay(10);
  
  Serial.print("\n");
}

int playing() {
  /* side-switching process
  *
  *  This process switches both directions of the relay repeatedly.
  *  @return: The currently active player
  */
  int r = currentPlayer;
  if (relayCurrentTime == 0) {
    digitalWrite(relayPin, HIGH);
    r = 0;
  } else if (relayCurrentTime == relayTime) {
    digitalWrite(relayPin, LOW);
    r = 1;
  }
  relayCurrentTime++;
  if (relayCurrentTime >= 2*relayTime) {
    relayCurrentTime = 0;
  }
  return r;
}

void shuffle(int shuffleTime) {
  /* Shuffling the LEDs
  */
  int rnd;
  if (shuffleCurrentTime == 0) {
    rnd = random(1000);
    digitalWrite(currentPin, LOW);
    if (rnd < LED1Prop) {
      currentPin = LED1Pin;
    } else if (rnd < LED2Prop) {
      currentPin = LED2Pin;
    } else if (rnd < LED3Prop) {
      currentPin = LED3Pin;
    } else {
      currentPin = LEDStatusPin;
    }
    digitalWrite(currentPin, HIGH);
  }
  shuffleCurrentTime++;
  if (shuffleCurrentTime >= 2*shuffleTime) {
    shuffleCurrentTime = 0;
  }
  Serial.print(shuffleCurrentTime);
  Serial.print("\n");
}

void counter() {
  shuffle(counterShuffleTime);
  counterCurrentTime++;
  if (counterCurrentTime >= 2*counterTimeGain) {
    counterShuffleTime -= counterTimeGain;
    counterCurrentTime = 0;
  }
  Serial.print(counterCurrentTime);
  Serial.print("\n");
}
