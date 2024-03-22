export function stt() {
  const recognition = new (window.SpeechRecognition ||
    window.webkitSpeechRecognition ||
    window.mozSpeechRecognition ||
    window.msSpeechRecognition)();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onresult = function (event) {
    const transcript = event.results[0][0].transcript;
    console.log("Transcript:", transcript);
    document.querySelector(".textarea-e85c3").value = transcript;
  };

  recognition.onerror = function (event) {
    console.error("Speech recognition error:", event.error);
    // Handle error, such as displaying a message to the user
  };

  recognition.start();
}
