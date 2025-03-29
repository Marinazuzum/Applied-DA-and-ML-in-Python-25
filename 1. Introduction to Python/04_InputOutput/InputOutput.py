def python_wiki_text():

    ### Enter your code here ###
    # Open the reversed file
    try:
        with open("1. Introduction to Python/04_InputOutput/PythonWikipedia_reversed.txt", "r") as infile:
            # Initialize counters
            line_count = 0
            word_count = 0
            char_count = 0
            char_with_spaces_count = 0
            
            # Open a new file to write the readable text
            with open("1. Introduction to Python/04_InputOutput/PythonWikipedia.txt", "w") as outfile:
                for line in infile:
                    line_count += 1
                    words = line.split()
                    word_count += len(words)
                    char_with_spaces_count += len(line)
                    char_count += sum(len(word) for word in words)

                    # Reverse the words in the line
                    reversed_line = ' '.join(words[::-1])

                    # Write the reversed line to the output file without trailing spaces
                    outfile.write(reversed_line + '\n')

            # Print the counts
            print(f"Lines: {line_count}")
            print(f"Words: {word_count}")
            print(f"Characters (without spaces): {char_count}")
            print(f"Characters (with spaces): {char_with_spaces_count}")

    except FileNotFoundError:
        print("The file PythonWikipedia_reversed.txt was not found.")



    ### End of your code ###


if __name__ == "__main__":
    python_wiki_text()