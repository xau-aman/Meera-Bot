# Meera

**Your personal AI mentor for coding, careers, and interviews — inside Discord.**

Meera is a Discord bot that combines coding practice, knowledge management, AI-powered Q&A, and gamification into one seamless experience. Built for developers who want structured preparation without scattered resources.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Commands](#commands)
- [Interactive Menu](#interactive-menu)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Training (Custom Model)](#training-custom-model)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Features

### Coding Practice
- Daily coding questions with difficulty-based XP rewards
- Topic-based practice (Arrays, Linked Lists, Stacks, Binary Search, DP, Sliding Window, Design)
- Solution submission with language tracking
- Hints system with spoiler tags

### Second Brain (Notes)
- Save notes with tags via modal popups
- Full-text search by title or tag
- Recent notes viewer
- Linked to your Discord identity

### AI Assistant
- Ask Meera anything about DSA, CS theory, interviews, or career advice
- Powered by Groq (Llama 3.3 70B) with a custom mentor personality
- Conversational, concise, and practical responses

### Gamification
- XP system: Easy (+10), Medium (+25), Hard (+50)
- Daily streak tracking with automatic detection
- Level progression with visual progress bars
- Server-wide leaderboard

### Button-Centric UI
- Mention Meera to open the interactive menu
- All features accessible via buttons and modals
- No need to memorize commands — just click

---

## Architecture

```
Discord User
    |
    v
Discord Gateway (discord.js v14)
    |
    ├── Button/Modal Interactions --> Feature Handlers
    ├── Slash Commands (legacy support)
    └── Mention Detection --> Interactive Menu
    |
    v
PostgreSQL (Prisma ORM)        Groq API (AI)
    |                              |
    ├── Users & XP                 └── Chat Completions
    ├── Questions                      (Llama 3.3 70B)
    ├── Submissions
    ├── Notes
    └── Daily Questions
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Node.js 18+ |
| Bot Framework | Discord.js v14 |
| Database | PostgreSQL |
| ORM | Prisma |
| AI | Groq (Llama 3.3 70B Versatile) |
| Custom Model | PyTorch (38M param transformer, trained from scratch) |

---

## Getting Started

### Prerequisites

- Node.js 18+
- PostgreSQL database
- Discord Bot Token — [Discord Developer Portal](https://discord.com/developers/applications)
- Groq API Key — [Groq Console](https://console.groq.com)

### Installation

```bash
git clone https://github.com/xau-aman/Meera-Bot.git
cd Meera-Bot
npm install
```

### Configuration

```bash
cp .env.example .env
```

Fill in your `.env`:

```env
DISCORD_TOKEN=your_discord_bot_token
DISCORD_CLIENT_ID=your_application_id
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://user:password@localhost:5432/meera
```

### Database Setup

```bash
npx prisma db push
npm run db:seed
```

### Register Slash Commands

```bash
npm run deploy-commands
```

### Discord Developer Portal Setup

1. Go to your application in the Developer Portal
2. Under Bot, enable these Privileged Gateway Intents:
   - MESSAGE CONTENT INTENT
   - SERVER MEMBERS INTENT
3. Generate an invite link with `bot` + `applications.commands` scopes
4. Add the bot to your server

### Run

```bash
npm start        # production
npm run dev      # development (auto-restart on file changes)
```

---

## Commands

| Command | Description |
|---------|-------------|
| `/question daily` | Get today's daily coding question |
| `/question topic <topic>` | Get a random question by topic |
| `/submit <id> <lang> <code>` | Submit a solution and earn XP |
| `/note add <title> <content> [tags]` | Save a note to your second brain |
| `/note search <query>` | Search notes by title or tag |
| `/ask <question>` | Ask Meera anything |
| `/progress` | View your XP, level, streak, and stats |
| `/leaderboard` | View the server's top coders |

---

## Interactive Menu

Mention Meera (`@Meera`) in any channel to open the interactive menu. From there:

- **Daily Question** — Get today's problem with hint and submit buttons
- **Pick a Topic** — Dropdown menu to select a topic, get a random question
- **My Progress** — View your stats with action buttons
- **Leaderboard** — See who's on top
- **My Notes** — Add, search, or browse recent notes via modals
- **Ask Meera** — Opens a text modal for AI-powered Q&A

Everything is navigable via buttons. No commands needed.

---

## Project Structure

```
Meera-Bot/
├── prisma/
│   ├── schema.prisma          # Database models
│   └── seed.js                # Starter DSA questions (10 problems)
├── src/
│   ├── commands/              # Slash command handlers
│   │   ├── ask.js
│   │   ├── leaderboard.js
│   │   ├── note.js
│   │   ├── progress.js
│   │   ├── question.js
│   │   └── submit.js
│   ├── events/
│   │   ├── buttonHandler.js   # All button/modal/select interactions
│   │   ├── mention.js         # Mention detection + menu
│   │   └── ready.js           # Bot startup
│   ├── lib/
│   │   ├── ai.js              # Groq client + Meera personality
│   │   ├── db.js              # Prisma client singleton
│   │   └── xp.js              # XP awards + streak logic
│   ├── deploy-commands.js     # Register slash commands with Discord
│   └── index.js               # Entry point
├── training/                  # Custom model training (PyTorch)
│   ├── data/raw/              # Training datasets (DSA, CS, interviews, etc.)
│   ├── scripts/
│   │   ├── model.py           # 38M param GPT architecture
│   │   ├── train.py           # Training loop
│   │   ├── inference.py       # Chat with trained model
│   │   ├── prepare_data.py    # Data preprocessing
│   │   └── train_tokenizer.py # BPE tokenizer training
│   └── requirements.txt
├── .env.example
├── .gitignore
├── package.json
└── README.md
```

---

## Deployment

### Railway (Recommended)

1. Push code to GitHub
2. Sign up at [railway.app](https://railway.app) with GitHub
3. Create new project from your GitHub repo
4. Add PostgreSQL database (click "+ New" then "Database")
5. Set environment variables:
   ```
   DISCORD_TOKEN=your_token
   DISCORD_CLIENT_ID=your_client_id
   GROQ_API_KEY=your_key
   DATABASE_URL=${{Postgres.DATABASE_URL}}
   ```
6. Set build command: `npm install && npx prisma db push && npm run db:seed`
7. Set start command: `npm start`
8. Deploy. Bot runs 24/7.

Auto-deploys on every `git push` to main.

---

## Training (Custom Model)

Meera includes a from-scratch transformer model (38M parameters) for experimentation.

### Setup

```bash
cd training
pip install -r requirements.txt
```

### Pipeline

```bash
python scripts/prepare_data.py      # Merge and split data
python scripts/train_tokenizer.py   # Train BPE tokenizer
python scripts/train.py             # Train the model
python scripts/inference.py         # Chat with it
```

### Model Specs

| Parameter | Value |
|-----------|-------|
| Parameters | 38M |
| Layers | 8 |
| Attention Heads | 8 |
| Embedding Dim | 512 |
| Feed-Forward Dim | 2048 |
| Max Sequence Length | 512 |
| Vocab Size | 8192 |
| Architecture | GPT-style decoder with SwiGLU + RMSNorm |

### Training Data Domains

- Data Structures and Algorithms (problems + explanations)
- CS Theory (OS, DBMS, Networking, OOP, SOLID)
- Interview Preparation (HR, behavioral, technical, system design)
- Resume and LinkedIn Optimization
- Cybersecurity Fundamentals (OWASP, common vulnerabilities)
- Meera Personality (conversational style)

---

## Roadmap

| Phase | Features | Status |
|-------|----------|--------|
| Phase 1 | Coding practice + Notes + XP | Done |
| Phase 2 | AI Assistant + Button UI | Done |
| Phase 3 | Resume and LinkedIn tools | Planned |
| Phase 4 | Cybersecurity + System Design modules | Planned |
| Phase 5 | Custom model integration (replace Groq) | In Progress |

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/something`)
3. Commit your changes
4. Push and open a Pull Request

---

## License

MIT

---

Built by [xau-aman](https://github.com/xau-aman)
