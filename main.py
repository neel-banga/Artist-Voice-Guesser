if __name__ == "__main__":


    # Simple terminal front end where the user is asked a few questions and the directories are sorted out

    import os
    # Import our audio.py file
    import audio 

    # Assigns the directory we're in right now to parent_dir
    PARENT_DIR_PATH = os.getcwd()

    while True:
        y_o_n = input(f'Would you like the contents of this program to be stored in (Y or N) {PARENT_DIR_PATH}:    ')
    

        if y_o_n.lower() == 'y':
            break

        # Get's an alternate path (if the user would like it relocated), makes sure it exists, then assigns it to that path
        elif y_o_n.lower() == 'n':
            possible_parent_dir = input('Where would you like the program to be located:    ')
            if os.path.exists(possible_parent_dir):
                print(f' \n Please move all the files in {PARENT_DIR_PATH} to {possible_parent_dir}. The data will be saved in {possible_parent_dir}. \n')
                parent_dir = possible_parent_dir
                break

            else:
                print('Directory does not exist, please try again')

        else:
            print('Option invalid. Please try again')



    # Now let's set this as an environment variable, so we can use this in audio.py and model.py
    #os.environ['PARENT_DIR'] = os.path.join(parent_dir, 'audio')
    

    print('Collecting Audio Data....')

    # Check if path doesn't exist, if so, get audio. If not, then don't
    
    #if os.path.exists(PARENT_DIR_PATH) == False:
    audio.start(os.path.join(PARENT_DIR_PATH, 'audio'))
    print('Audio Collection Done.')
