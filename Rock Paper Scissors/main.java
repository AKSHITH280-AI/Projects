import java.util.Random;
import java.util.Scanner;

public class RockPaperScissors {
    public static void main(String[] args) {
        
        System.out.println("Winning rules of the game ROCK PAPER SCISSORS are:\n"
                + "Rock vs Paper -> Paper wins \n"
                + "Rock vs Scissors -> Rock wins \n"
                + "Paper vs Scissors -> Scissor wins \n");

        Scanner sc = new Scanner(System.in);
        Random rand = new Random();
        String playAgain;

        do {
            System.out.println("Enter your choice: \n 1 - Rock \n 2 - Paper \n 3 - Scissors \n");
            int choice = sc.nextInt();

            
            while (choice > 3 || choice < 1) {
                System.out.println("Enter a valid choice please: ");
                choice = sc.nextInt();
            }

            String choiceName;
            if (choice == 1) {
                choiceName = "Rock";
            } else if (choice == 2) {
                choiceName = "Paper";
            } else {
                choiceName = "Scissors";
            }

            System.out.println("User choice is: " + choiceName);
            System.out.println("Now it's Computer's Turn...");

            int compChoice = rand.nextInt(3) + 1;
            String compChoiceName;
            if (compChoice == 1) {
                compChoiceName = "Rock";
            } else if (compChoice == 2) {
                compChoiceName = "Paper";
            } else {
                compChoiceName = "Scissors";
            }

            System.out.println("Computer choice is: " + compChoiceName);
            System.out.println(choiceName + " Vs " + compChoiceName);

            String result;
            if (choice == compChoice) {
                result = "DRAW";
                System.out.println("<== It's a tie! ==>");

            } else if ((choice == 1 && compChoice == 2) || (choice == 2 && compChoice == 1)) {
                result = "Paper";
                System.out.println("Paper wins =>");
            } else if ((choice == 1 && compChoice == 3) || (choice == 3 && compChoice == 1)) {
                result = "Rock";
                System.out.println("Rock wins =>");
            } else {
                result = "Scissors";
                System.out.println("Scissors wins =>");
            }

            if (!result.equals("DRAW")) {
                if (result.equals(choiceName)) {
                    System.out.println("<== User wins! ==>");
                } else {
                    System.out.println("<== Computer wins! ==>");
                }
            }

            System.out.println("Do you want to play again? (Y/N)");
            playAgain = sc.next().toLowerCase();
        } while (playAgain.equals("y"));

        System.out.println("Thanks for playing!");
        sc.close();
    }
}

