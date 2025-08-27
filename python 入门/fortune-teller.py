import random
import time

def fortune_teller():
    print("🔮 Welcome to the Python Fortune Teller 🔮")
    print("I will reveal what your future holds!")
    
    # Ask for the user's name
    name = input("\nWhat is your name? ")
    print(f"\nHello {name}, I'm looking into your future...")
    
    # Add a dramatic pause with dots appearing
    for _ in range(3):
        time.sleep(0.7)
        print(".", end="", flush=True)
    print("\n")
    
    # List of possible fortunes
    fortunes = [
        "You will find unexpected joy in something small today.",
        "A surprising friendship is about to blossom in your life.",
        "Your creativity will lead you to success this month.",
        "A small change in your routine will bring great happiness.",
        "You will soon master a skill you've been practicing.",
        "An exciting journey awaits you in the near future.",
        "Someone is thinking fondly of you right now.",
        "Your kindness will return to you tenfold.",
        "A pleasant surprise is on its way to you.",
        "Your hard work is about to pay off in an unexpected way."
    ]
    
    # Select a random fortune
    your_fortune = random.choice(fortunes)
    
    # Display the fortune
    print(f"✨ Your Fortune: {your_fortune} ✨")
    
    # Ask if they want another fortune
    while True:
        another = input("\nWould you like another fortune? (yes/no): ").lower()
        if another == "yes" or another == "y":
            print("\nLooking deeper into your future...")
            for _ in range(3):
                time.sleep(0.7)
                print(".", end="", flush=True)
            print("\n")
            
            # Get a different fortune than the previous one
            new_fortune = random.choice([f for f in fortunes if f != your_fortune])
            your_fortune = new_fortune
            print(f"✨ Your Fortune: {your_fortune} ✨")
        elif another == "no" or another == "n":
            print("\nThank you for visiting the Python Fortune Teller!")
            print("May your future be as bright as your curiosity! ✨")
            break
        else:
            print("Please answer with 'yes' or 'no'.")

if __name__ == "__main__":
    fortune_teller()
