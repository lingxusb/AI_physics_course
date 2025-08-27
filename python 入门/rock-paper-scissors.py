import random
import time

def rock_paper_scissors():
    """A simple Rock Paper Scissors game against the computer"""
    
    # Initialize scores
    player_score = 0
    computer_score = 0
    ties = 0
    
    # Define the possible choices
    choices = ["rock", "paper", "scissors"]
    
    # Create a dictionary to show which choice beats what
    beats = {
        "rock": "scissors",
        "paper": "rock",
        "scissors": "paper"
    }
    
    # Emoji representations for visual appeal
    emojis = {
        "rock": "🪨",
        "paper": "📄",
        "scissors": "✂️",
        "win": "🎉",
        "lose": "😢",
        "tie": "🤝"
    }
    
    print("\n===== ROCK PAPER SCISSORS =====")
    print("First to 3 wins is the champion!")
    
    # Main game loop
    while player_score < 3 and computer_score < 3:
        # Get player's choice
        print("\n" + "="*30)
        print(f"Score: You {player_score} - {computer_score} Computer (Ties: {ties})")
        
        player_choice = input("\nChoose rock, paper, or scissors (or 'quit' to exit): ").lower()
        
        # Check if player wants to quit
        if player_choice == 'quit':
            print("\nThanks for playing!")
            return
        
        # Validate player's choice
        if player_choice not in choices:
            print("Invalid choice! Please choose rock, paper, or scissors.")
            continue
        
        # Get computer's choice
        computer_choice = random.choice(choices)
        
        # Add suspense with a countdown
        print("\nRock...")
        time.sleep(0.5)
        print("Paper...")
        time.sleep(0.5)
        print("Scissors...")
        time.sleep(0.5)
        print("Shoot!\n")
        
        # Show choices
        print(f"You chose: {player_choice} {emojis[player_choice]}")
        print(f"Computer chose: {computer_choice} {emojis[computer_choice]}")
        
        # Determine the winner
        if player_choice == computer_choice:
            print(f"\nIt's a tie! {emojis['tie']}")
            ties += 1
        elif beats[player_choice] == computer_choice:
            print(f"\nYou win this round! {emojis['win']}")
            player_score += 1
        else:
            print(f"\nComputer wins this round! {emojis['lose']}")
            computer_score += 1
    
    # Game over - show final result
    print("\n" + "="*30)
    print("GAME OVER!")
    
    if player_score > computer_score:
        print(f"\nCongratulations! You are the champion! {emojis['win']}")
    else:
        print(f"\nThe computer has defeated you! Better luck next time! {emojis['lose']}")
    
    print(f"\nFinal Score: You {player_score} - {computer_score} Computer (Ties: {ties})")
    
    # Ask to play again
    play_again = input("\nWould you like to play again? (yes/no): ").lower()
    if play_again == "yes" or play_again == "y":
        rock_paper_scissors()
    else:
        print("\nThanks for playing! Goodbye!")

if __name__ == "__main__":
    rock_paper_scissors()
