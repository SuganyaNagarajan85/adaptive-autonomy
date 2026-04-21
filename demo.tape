# VHS tape — generates assets/demo.gif
# Install: brew install vhs
# Run:     vhs demo.tape

Output assets/demo.gif
Output assets/demo.mp4

Set Shell "bash"
Set FontSize 14
Set Width 1100
Set Height 700
Set Theme "Molokai"
Set Padding 20
Set PlaybackSpeed 1.5

# Start
Type "python3 demo.py --fast"
Sleep 500ms
Enter

# Wait for scenarios to run
Sleep 20s

# Pause at final output so viewers can read it
Sleep 3s
