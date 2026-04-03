/**
 * Neural Network Background Animation
 * Creates a dynamic web of nodes and connections
 */
const canvas = document.getElementById('neural-bg');
const ctx = canvas.getContext('2d');

let particles = [];
const properties = {
    bgColor: '#0f111a',
    particleColor: 'rgba(59, 130, 246, 0.5)',
    lineColor: 'rgba(109, 40, 217, 0.15)',
    particleRadius: 2,
    particleCount: 80,
    particleMaxVelocity: 0.5,
    lineLength: 150,
    particleLife: 15
};

window.onresize = function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
};

class Particle {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.velocityX = Math.random() * (properties.particleMaxVelocity * 2) - properties.particleMaxVelocity;
        this.velocityY = Math.random() * (properties.particleMaxVelocity * 2) - properties.particleMaxVelocity;
    }

    rePosition() {
        if (this.x > canvas.width || this.x < 0) this.velocityX *= -1;
        if (this.y > canvas.height || this.y < 0) this.velocityY *= -1;
        this.x += this.velocityX;
        this.y += this.velocityY;
    }

    draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, properties.particleRadius, 0, Math.PI * 2);
        ctx.closePath();
        ctx.fillStyle = properties.particleColor;
        ctx.fill();
    }
}

function drawLines() {
    let x1, y1, x2, y2, length, opacity;
    for (let i in particles) {
        for (let j in particles) {
            x1 = particles[i].x;
            y1 = particles[i].y;
            x2 = particles[j].x;
            y2 = particles[j].y;
            length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
            if (length < properties.lineLength) {
                opacity = 1 - length / properties.lineLength;
                ctx.lineWidth = '0.5';
                ctx.strokeStyle = `rgba(109, 40, 217, ${opacity * 0.3})`;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.closePath();
                ctx.stroke();
            }
        }
    }
}

function init() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    particles = [];
    for (let i = 0; i < properties.particleCount; i++) {
        particles.push(new Particle());
    }
    loop();
}

function loop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i in particles) {
        particles[i].rePosition();
        particles[i].draw();
    }
    drawLines();
    requestAnimationFrame(loop);
}

init();
