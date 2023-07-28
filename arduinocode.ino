/*........  EMF Detector	*/
/*........	EMF Detector Using Arduino	*/
/*........	Manik Garg - 2K20/EE/159	*/




/*........	Arduino Code for EMF Detector (Multi LED)	*/


const int analogPin = 0;	// Analog Input Pin const int ledCount = 10;	// No of LEDs

int ledPins[] = {
1, 2, 3, 4, 5, 6, 7, 8, 9, 10
};	/* An array of pin numbers to which LEDs are connected*/


void setup() {
/* Loop over the ledPins array and set them all to OUTPUT*/ for(int thisLed = 0; thisLed < ledCount; thisLed++){
pinMode(ledPins[thisLed], OUTPUT);
}
}






void loop() {
/* Reading the Input Signal*/
int sensorReading = analogRead(analogPin);


/* Map the result to a range from 0 to 10*/
int ledLevel = map(sensorReading, 0, 1023, 0, ledCount);


for(int thisLed = 0; thisLed < ledCount; thisLed++){


/* If the LED pin's index is less than ledLevel, then turn the LED ON for the pin*/
if(thisLed < ledLevel){ digitalWrite(ledPins[thisLed], HIGH);
}


/* Turn OFF all the pins higher than the ledLevel*/ else{
digitalWrite(ledPins[thisLed], LOW);
}
}


}
