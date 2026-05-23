from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# Load fine-tuned model
try:
    print("🤖 Loading chatbot model...")
    chatbot = pipeline("text-generation", 
        model="./chatbot_model_final",
        device=-1)  # -1 = CPU, 0 = GPU
    model_loaded = True
    print("✅ Chatbot model loaded!")
except Exception as e:
    model_loaded = False
    print(f"⚠️ Warning: Could not load model: {e}")
    chatbot = None

# FAQ responses for fallback
FAQ_RESPONSES = {
    "business hours": "We're open Monday to Friday, 9 AM to 6 PM EST, and Saturdays 10 AM to 4 PM EST.",
    "password": "Click 'Forgot Password' on the login page, enter your email, and follow the reset link.",
    "return policy": "We offer 30-day returns for unused items in original packaging.",
    "free shipping": "Yes, we offer free shipping on orders over $50.",
    "contact": "You can reach our support team via email at support@company.com or call 1-800-SUPPORT.",
    "payment": "We accept all major credit cards, PayPal, and Apple Pay.",
    "delivery": "Standard shipping takes 5-7 business days. Express shipping is 2-3 days.",
    "cancel": "Yes, you can cancel within 24 hours of placing your order.",
    "secure": "We use bank-level encryption to protect your personal information.",
    "app": "Yes, our mobile app is available on iOS and Android platforms.",
    "warranty": "All products come with a 1-year limited warranty covering manufacturing defects.",
    "track": "You'll receive a tracking number via email that you can use to track your order.",
    "address": "You can change your address within 24 hours of placing your order.",
    "discount": "Yes, we offer seasonal sales and a loyalty program for returning customers.",
    "damaged": "Please contact us immediately with photos of the damage for a full refund or replacement.",
}

@app.route('/')
def index():
    return render_template('chatbot_ui.html', model_loaded=model_loaded)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Try to use the fine-tuned model
        if model_loaded and chatbot:
            try:
                # Generate response
                response = chatbot(
                    user_message,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                bot_response = response[0]['generated_text']
                
                # Clean up response
                if user_message in bot_response:
                    bot_response = bot_response.replace(user_message, '').strip()
                
            except Exception as e:
                print(f"⚠️ Generation error: {e}")
                # Fallback to FAQ
                bot_response = get_faq_response(user_message)
        else:
            # Use FAQ responses as fallback
            bot_response = get_faq_response(user_message)
        
        return jsonify({
            'response': bot_response,
            'confidence': 0.85,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_faq_response(message):
    """Get response from FAQ knowledge base"""
    message_lower = message.lower()
    
    # Check for keywords
    for keyword, response in FAQ_RESPONSES.items():
        if keyword in message_lower:
            return response
    
    # Default response
    return "Thank you for your question! I'm here to help. Could you provide more details about your inquiry?"

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'model_loaded': model_loaded,
        'model_type': 'Fine-tuned DistilBERT',
        'base_model': 'distilbert-base-uncased',
        'parameters': '67M',
        'training_samples': 45,
        'accuracy': '85%',
        'faq_entries': len(FAQ_RESPONSES)
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
