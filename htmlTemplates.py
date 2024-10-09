# CSS for styling the chat messages
css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    align-items: center;
}

.chat-message.user {
    justify-content: flex-end;
    background-color: #2b313e;
}

.chat-message.bot {
    justify-content: flex-start;
    background-color: #475063;
}

.chat-message .avatar {
    width: 60px;
    height: 60px;
    margin-right: 1rem;
}

.chat-message.user .avatar {
    margin-left: 1rem;
    margin-right: 0;
}

.chat-message .avatar img {
    max-width: 100%;
    max-height: 100%;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    max-width: 80%;
    padding: 1rem;
    color: #fff;
    border-radius: 0.5rem;
    line-height: 1.5;
    word-wrap: break-word;
}
</style>
'''

# HTML template for bot messages (left-aligned)
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/6134/6134346.png" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

# HTML template for user messages (right-aligned)
user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
    <div class="avatar">
        <img src="https://png.pngtree.com/png-vector/20190321/ourmid/pngtree-vector-users-icon-png-image_856952.jpg" alt="User Avatar">
    </div>    
</div>
'''
