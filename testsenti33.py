import customtkinter as ctk
from tkinter import filedialog, messagebox, Menu
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import nltk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize the Sentiment Intensity Analyzer
sid = SentimentIntensityAnalyzer()

# Define emoji responses
emoji_responses = {
    'positive': ['😊', '😄', '🌟'],
    'negative': ['😔', '😞', '💔'],
    'neutral': ['😐', '🤔', '😶']
}

# Colors for sentiments
emoji_colors = {
    'positive': 'green',
    'negative': 'red',
    'neutral': 'gray'
}

class SentimentDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Detector")
        self.root.geometry("800x600")
        self.root.configure(bg="#2E2E2E")

        self.history = []

        self.create_menu()
        self.create_layout()
        self.create_intro_screen()

    def create_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.create_prompt_screen)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_layout(self):
        # Frame for history
        history_label = ctk.CTkLabel(self.root, text="History", font=("Helvetica", 14), text_color="white")
        history_label.pack(fill="x", pady=10)

        self.history_frame = ctk.CTkFrame(self.root, width=200)
        self.history_frame.pack(side="left", fill="y")

        self.history_listbox = ctk.CTkTextbox(self.history_frame, width=200)
        self.history_listbox.pack(expand=True, fill="both", pady=10, padx=10)

        # Frame for main content
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(side="right", expand=True, fill="both")

    def create_intro_screen(self):
        if hasattr(self, 'current_screen') and self.current_screen:
            self.current_screen.destroy()

        self.current_screen = ctk.CTkFrame(self.main_frame)
        self.current_screen.pack(expand=True, fill="both")

        label = ctk.CTkLabel(self.current_screen, text="Welcome to Sentiment Detector", font=("Helvetica", 24), text_color="white")
        label.pack(pady=50)

        start_button = ctk.CTkButton(self.current_screen, text="Start", font=("Helvetica", 18), command=self.create_prompt_screen)
        start_button.pack()

    def create_prompt_screen(self):
        if hasattr(self, 'current_screen') and self.current_screen:
            self.current_screen.destroy()

        self.current_screen = ctk.CTkFrame(self.main_frame)
        self.current_screen.pack(expand=True, fill="both")

        label = ctk.CTkLabel(self.current_screen, text="Enter your prompt to detect sentiment:", font=("Helvetica", 18), text_color="white")
        label.pack(pady=20)

        # Creating a text area box with grey background color and placeholder text
        self.prompt_entry = ctk.CTkTextbox(self.current_screen, height=10, width=60, font=("Helvetica", 14), fg_color="#333333", bg="#CCCCCC", text_color="white")
        self.prompt_entry.insert("1.0", "Enter the prompt...")
        self.prompt_entry.pack(pady=20)

        analyze_button = ctk.CTkButton(self.current_screen, text="Analyze Sentiment", font=("Helvetica", 16), command=self.analyze_sentiment)
        analyze_button.pack()

    def analyze_sentiment(self):
        text = self.prompt_entry.get("1.0", "end").strip()
        if text:
            scores = sid.polarity_scores(text)
            compound_score = scores['compound']

            if compound_score >= 0.05:
                sentiment = "positive"
            elif compound_score <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            self.history.append((text, sentiment))
            self.update_history()
            self.display_sentiment_response(sentiment, scores)

    def display_sentiment_response(self, sentiment, scores):
        if hasattr(self, 'current_screen') and self.current_screen:
            self.current_screen.destroy()

        self.current_screen = ctk.CTkFrame(self.main_frame)
        self.current_screen.pack(expand=True, fill="both")

        label = ctk.CTkLabel(self.current_screen, text="Sentiment Analysis Result", font=("Helvetica", 24), text_color="white")
        label.pack(pady=20)

        emoji_label = ctk.CTkLabel(self.current_screen, text=random.choice(emoji_responses[sentiment]), font=("Arial", 50), text_color=emoji_colors[sentiment])
        emoji_label.pack(pady=20)

        sentiment_label = ctk.CTkLabel(self.current_screen, text=f"Sentiment: {sentiment.capitalize()}", font=("Helvetica", 18), text_color="white")
        sentiment_label.pack(pady=10)

        self.plot_sentiment_pie_chart(scores)

        # Analyze another prompt button
        new_prompt_button = ctk.CTkButton(self.current_screen, text="Analyze Another Prompt", font=("Helvetica", 16), command=self.create_prompt_screen)
        new_prompt_button.pack(pady=10)

        # New button
        new_button = ctk.CTkButton(self.current_screen, text="New", font=("Helvetica", 16), command=self.create_prompt_screen)
        new_button.pack(pady=10)

        # Exit button
        exit_button = ctk.CTkButton(self.current_screen, text="Exit", font=("Helvetica", 16), command=self.root.quit)
        exit_button.pack(pady=10)

    def plot_sentiment_pie_chart(self, scores):
        fig, ax = plt.subplots(facecolor='#2E2E2E')
        fig.patch.set_facecolor('#2E2E2E')

        sentiments = ['Positive', 'Neutral', 'Negative']
        scores_list = [scores['pos'], scores['neu'], scores['neg']]
        colors = ['green', 'gray', 'red']

        ax.pie(scores_list, labels=sentiments, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'color':'white'})
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        canvas = FigureCanvasTkAgg(fig, master=self.current_screen)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=20)

    def update_history(self):
        self.history_listbox.delete("1.0", "end")
        for prompt, sentiment in self.history:
            self.history_listbox.insert("end", f"{prompt[:20]}...: {sentiment}\n")

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                self.prompt_entry.delete('1.0', "end")
                self.prompt_entry.insert("end", content)

    def save_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, 'w') as file:
                content = self.prompt_entry.get("1.0", "end").strip()
                file.write(content)

    def show_about(self):
        messagebox.showinfo("About", "Sentiment Detector\nVersion 1.0\nDeveloped by OpenAI")

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")  # Modes: "System" (default), "Light", "Dark"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "dark-blue", "green"

    root = ctk.CTk()  # create CTk window
    app = SentimentDetectorApp(root)
    root.mainloop()
