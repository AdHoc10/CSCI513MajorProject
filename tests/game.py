from simfire.game.game import Game
import pygame

def main():
    # Example parameters
    screen_size = (800, 600)  # Arbitrary pixel dimensions (width, height)
    rescale_factor = 1  # 1 means no rescaling, 2 would double the size, etc.
    
    # Create game instance
    game = Game(
        screen_size=screen_size,
        rescale_factor=rescale_factor,
        headless=False,  # Show the window
        record=False,    # Don't record
        show_wind_magnitude=True,  # Show wind magnitude visualization
        show_wind_direction=True,  # Show wind direction visualization
    )
    
    # Keep window open until user closes it
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    # Clean up
    game.quit()

if __name__ == "__main__":
    main()