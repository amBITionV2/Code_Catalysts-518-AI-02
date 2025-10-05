class KCETChatbot {
    constructor() {
        this.chatIcon = document.querySelector('.chatbot-icon');
        this.chatContainer = document.querySelector('.chat-container');
        this.chatMessages = document.querySelector('.chat-messages');
        this.chatInput = document.querySelector('.chat-input input');
        this.sendButton = document.querySelector('.chat-input button');
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        this.chatIcon.addEventListener('click', () => this.toggleChat());
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
    }

    toggleChat() {
        this.chatContainer.style.display = 
            this.chatContainer.style.display === 'none' ? 'flex' : 'none';
        if (this.chatContainer.style.display === 'flex') {
            this.chatInput.focus();
        }
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        try {
            // Disable input while processing
            this.sendButton.disabled = true;
            this.chatInput.disabled = true;

            // Add user message
            this.addMessage(message, 'user');
            this.chatInput.value = '';

            // Send request to server
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            // Parse response exactly once
            let responseData;
            try {
                responseData = await response.json();
            } catch (err) {
                throw new Error('Failed to parse server response');
            }

            // Check for errors
            if (!response.ok || responseData.error) {
                throw new Error(responseData.error || 'Server error');
            }

            // Display bot response
            if (responseData.response) {
                this.addMessage(responseData.response, 'bot');
            } else {
                throw new Error('Invalid response format');
            }

        } catch (error) {
            console.error('Chat error:', error);
            let errorMessage = 'An error occurred. Please try again.';
            
            if (error.message.includes('Failed to fetch')) {
                errorMessage = 'Network error. Please check your internet connection.';
            } else if (error.message.includes('API key')) {
                errorMessage = 'Service configuration error. Please contact support.';
            } else if (error.message) {
                errorMessage = error.message;
            }

            this.addMessage(errorMessage, 'bot error');
        } finally {
            // Always re-enable input
            this.sendButton.disabled = false;
            this.chatInput.disabled = false;
            this.chatInput.focus();
        }
    }

    addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        
        switch(type) {
            case 'user':
                messageDiv.classList.add('user-message');
                break;
            case 'bot error':
                messageDiv.classList.add('bot-message', 'bot-error');
                break;
            default:
                messageDiv.classList.add('bot-message');
        }
        
        messageDiv.textContent = text;
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new KCETChatbot();
});
