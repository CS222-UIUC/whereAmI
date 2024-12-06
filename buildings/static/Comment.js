const comments = [];

// Define a list of usernames and their corresponding icons
const anonymousUsers = [
    { name: 'Anonymous Alma Mater', icon: '/static/Alma_mater.png' },
    { name: 'Anonymous Bob', icon: '/static/Bob.png' },
    { name: 'Anonymous Lincoln', icon: '/static/Lincoln.png' },
    { name: 'Anonymous Quinn', icon: '/static/Quinn.png' },
    { name: 'Anonymous Red Grange', icon: '/static/Red.png' }
];


// function renderComments() {
//     event.preventDefault();
//     const content = document.getElementById('contentOfTextarea').value;
//     comments.push(content);

//     let commentsHTML = '';
    
//     comments.forEach((comment) => {
//         commentsHTML += `
//             <div class="comments">
//                 <strong class="username"> Anonymous </strong>
//                 <p class="usercomment">${comment}</p>
//             </div>
//         `;
//     })
//     console.log(commentsHTML);
//     document.querySelector('.js-comment-section').
//     innerHTML = commentsHTML;
//     document.getElementById('contentOfTextarea').value = '';
// }

function renderComments() {
    console.log("renderComments function triggered");
    event.preventDefault();

    const content = document.getElementById('contentOfTextarea').value;
    console.log("Comment content:", content);

    if (!content.trim()) {
        alert("Comment cannot be empty.");
        return;
    }

    // Select a random anonymous user
    const randomUser = anonymousUsers[Math.floor(Math.random() * anonymousUsers.length)];
    console.log("Selected user:", randomUser);

    // Add the comment to the comments array
    comments.push({ content, user: randomUser });
    console.log("Updated comments array:", comments);

    let commentsHTML = '';

    // Generate HTML for all comments
    comments.forEach((comment) => {
        commentsHTML += `
            <div class="comments">
                <div class="comment-header">
                    <img src="${comment.user.icon}" alt="${comment.user.name}" class="user-icon" />
                    <strong class="username">${comment.user.name}</strong>
                </div>
                <p class="usercomment">${comment.content}</p>
            </div>
        `;
    });

    console.log("Generated comments HTML:", commentsHTML);

    // Update the comments section
    document.querySelector('.js-comment-section').innerHTML = commentsHTML;

    // Clear the textarea
    document.getElementById('contentOfTextarea').value = '';
}
