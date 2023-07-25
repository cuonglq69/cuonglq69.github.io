// Function to show a notice with a given id and text
function showNotice(noticeId, text) {
    const notice = document.getElementById(noticeId);
    notice.innerText = text;
    notice.style.display = 'block';
}

// Function to hide a notice with a given id
function hideNotice(noticeId) {
    const notice = document.getElementById(noticeId);
    notice.style.display = 'none';
}

// Function to generate music
function generateMusic() {
    const generateButton = document.getElementById('generateButton');
    generateButton.disabled = true;

    showNotice('generateNotice', 'Creating Music...');

    // Send a request to the '/generate' endpoint to generate music
    fetch('/generate')
        .then(response => response.text())
        .then(() => {
            const musicPlayer = document.getElementById('musicPlayer');
            const myStaffVisual = document.getElementById('myStaffVisual');
            const myPianoRollVisualizer = document.getElementById('myPianoRollVisualizer');

            // Update the source of the music player to the generated song
            newSource = "/static/song2.mid";
            musicPlayer.src = newSource ; // Add timestamp to force reload
            myPianoRollVisualizer.src = newSource ;
            myStaffVisual.src = newSource ;

            musicPlayer.style.display = 'block'; // Show the music player
            myStaffVisual.style.display = 'block'; // Show the staff visualizer
            myPianoRollVisualizer.style.display = 'block'; // Show the piano roll visualizer

            hideNotice('generateNotice');
            showNotice('generatedNotice', 'Your music is available');

            generateButton.disabled = false;
        });
}

// Add an event listener to the generateButton
const generateButton = document.getElementById('generateButton');
generateButton.addEventListener('click', function () {
    const generatedNotice = document.getElementById('generatedNotice');
    const musicPlayer = document.getElementById('musicPlayer');
    const myStaffVisual = document.getElementById('myStaffVisual');
    const myPianoRollVisualizer = document.getElementById('myPianoRollVisualizer');

    if (generatedNotice.style.display === 'block') {
        hideNotice('generatedNotice');
        musicPlayer.style.display = 'none'; // Hide the music player
        myStaffVisual.style.display = 'none'; // Hide the staff visualizer
        myPianoRollVisualizer.style.display = 'none'; // Hide the piano roll visualizer
    }

    generateMusic();
});

// Hide the music player and visualizers initially
const musicPlayer = document.getElementById('musicPlayer');
musicPlayer.style.display = 'none';
const myStaffVisual = document.getElementById('myStaffVisual');
myStaffVisual.style.display = 'none';
const myPianoRollVisualizer = document.getElementById('myPianoRollVisualizer');
myPianoRollVisualizer.style.display = 'none';
