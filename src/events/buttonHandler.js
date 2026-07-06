const {
  EmbedBuilder,
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
  ModalBuilder,
  TextInputBuilder,
  TextInputStyle,
  StringSelectMenuBuilder,
} = require("discord.js");
const prisma = require("../lib/db");
const { ensureUser, awardXP } = require("../lib/xp");
const { chat } = require("../lib/ai");
const { joinVC, leaveVC, isInVC } = require("../voice/voiceHandler");

module.exports = {
  name: "interactionCreate",
  async execute(interaction) {
    // Slash commands
    if (interaction.isChatInputCommand()) {
      const command = interaction.client.commands.get(interaction.commandName);
      if (!command) return;
      try {
        await command.execute(interaction);
      } catch (error) {
        console.error(`Error executing ${interaction.commandName}:`, error);
        const msg = { content: "Oops, something broke! Try again? 💜", ephemeral: true };
        if (interaction.replied || interaction.deferred) await interaction.followUp(msg);
        else await interaction.reply(msg);
      }
      return;
    }

    // Button clicks
    if (interaction.isButton()) {
      await handleButton(interaction);
      return;
    }

    // Select menus
    if (interaction.isStringSelectMenu()) {
      await handleSelect(interaction);
      return;
    }

    // Modals
    if (interaction.isModalSubmit()) {
      await handleModal(interaction);
      return;
    }
  },
};

async function handleButton(i) {
  const id = i.customId;

  // ── Main Menu ──
  if (id === "menu_home") {
    const embed = new EmbedBuilder()
      .setTitle("✨ Hey, I'm Meera!")
      .setDescription(
        `Hey **${i.user.displayName}**! I'm your personal AI mentor for coding, careers & interviews.\n\nWhat do you wanna do today? Pick something below 👇`
      )
      .setColor(0x7c3aed)
      .setFooter({ text: "Meera — Smart, sharp & always here for you 💜" });

    const row1 = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId("menu_daily").setLabel("📅 Daily Question").setStyle(ButtonStyle.Primary),
      new ButtonBuilder().setCustomId("menu_topics").setLabel("📚 Pick a Topic").setStyle(ButtonStyle.Primary),
      new ButtonBuilder().setCustomId("menu_progress").setLabel("📊 My Progress").setStyle(ButtonStyle.Success),
      new ButtonBuilder().setCustomId("menu_leaderboard").setLabel("🏆 Leaderboard").setStyle(ButtonStyle.Success),
    );
    const row2 = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId("menu_notes").setLabel("📝 My Notes").setStyle(ButtonStyle.Secondary),
      new ButtonBuilder().setCustomId("menu_ask").setLabel("🧠 Ask Meera").setStyle(ButtonStyle.Danger),
      new ButtonBuilder().setCustomId("menu_voice_join").setLabel("🎙️ Join VC").setStyle(ButtonStyle.Secondary),
    );
    return i.update({ embeds: [embed], components: [row1, row2] });
  }

  // ── Daily Question ──
  if (id === "menu_daily") {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    let daily = await prisma.dailyQuestion.findUnique({ where: { date: today } });
    if (!daily) {
      const count = await prisma.question.count();
      const dayOfYear = Math.floor((today - new Date(today.getFullYear(), 0, 0)) / 86400000);
      daily = await prisma.dailyQuestion.create({
        data: { questionId: (dayOfYear % count) + 1, date: today },
      });
    }

    const q = await prisma.question.findUnique({ where: { id: daily.questionId } });
    const embed = buildQuestionEmbed(q, "📅 Daily Question");

    const row = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId(`hint_${q.id}`).setLabel("💡 Show Hint").setStyle(ButtonStyle.Secondary),
      new ButtonBuilder().setCustomId(`submit_${q.id}`).setLabel("✍️ Submit Solution").setStyle(ButtonStyle.Success),
      new ButtonBuilder().setCustomId("menu_home").setLabel("🏠 Back").setStyle(ButtonStyle.Danger),
    );
    return i.update({ embeds: [embed], components: [row] });
  }

  // ── Topic Picker ──
  if (id === "menu_topics") {
    const embed = new EmbedBuilder()
      .setTitle("📚 Pick a Topic")
      .setDescription("Choose a topic and I'll throw a random question at you!")
      .setColor(0x3b82f6);

    const menu = new StringSelectMenuBuilder()
      .setCustomId("select_topic")
      .setPlaceholder("Choose a topic...")
      .addOptions(
        { label: "Arrays", value: "Arrays", emoji: "📊" },
        { label: "Linked List", value: "Linked List", emoji: "🔗" },
        { label: "Stack", value: "Stack", emoji: "📚" },
        { label: "Binary Search", value: "Binary Search", emoji: "🔍" },
        { label: "Dynamic Programming", value: "Dynamic Programming", emoji: "🧩" },
        { label: "Sliding Window", value: "Sliding Window", emoji: "🪟" },
        { label: "Design", value: "Design", emoji: "🏗️" },
      );

    const row1 = new ActionRowBuilder().addComponents(menu);
    const row2 = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId("menu_home").setLabel("🏠 Back").setStyle(ButtonStyle.Danger),
    );
    return i.update({ embeds: [embed], components: [row1, row2] });
  }

  // ── Progress ──
  if (id === "menu_progress") {
    const user = await ensureUser(i.user.id);
    const submissions = await prisma.submission.count({ where: { userId: i.user.id } });
    const level = Math.floor(user.xp / 100) + 1;
    const xpToNext = 100 - (user.xp % 100);
    const bar = "█".repeat(Math.floor((user.xp % 100) / 10)) + "░".repeat(10 - Math.floor((user.xp % 100) / 10));

    const embed = new EmbedBuilder()
      .setTitle(`📊 ${i.user.displayName}'s Progress`)
      .setDescription(`Looking good! Here's where you stand 👇`)
      .addFields(
        { name: "⭐ Level", value: `${level}`, inline: true },
        { name: "✨ Total XP", value: `${user.xp}`, inline: true },
        { name: "🔥 Streak", value: `${user.streak} day(s)`, inline: true },
        { name: "🧠 Problems Solved", value: `${submissions}`, inline: true },
        { name: "📈 Next Level", value: `${bar} (${xpToNext} XP to go)` },
      )
      .setColor(0x7c3aed)
      .setFooter({ text: "Keep pushing, you're doing amazing! 💜" });

    const row = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId("menu_daily").setLabel("📅 Solve a Question").setStyle(ButtonStyle.Primary),
      new ButtonBuilder().setCustomId("menu_home").setLabel("🏠 Back").setStyle(ButtonStyle.Danger),
    );
    return i.update({ embeds: [embed], components: [row] });
  }

  // ── Leaderboard ──
  if (id === "menu_leaderboard") {
    const users = await prisma.user.findMany({ orderBy: { xp: "desc" }, take: 10 });

    let desc = "No one on the board yet. Be the first! 🚀";
    if (users.length) {
      const medals = ["🥇", "🥈", "🥉"];
      const lines = await Promise.all(
        users.map(async (u, idx) => {
          const member = await i.guild.members.fetch(u.id).catch(() => null);
          const name = member?.displayName || "Unknown";
          const prefix = medals[idx] || `**${idx + 1}.**`;
          return `${prefix} ${name} — **${u.xp} XP** (🔥 ${u.streak})`;
        })
      );
      desc = lines.join("\n");
    }

    const embed = new EmbedBuilder()
      .setTitle("🏆 Leaderboard")
      .setDescription(desc)
      .setColor(0xf59e0b)
      .setFooter({ text: "Grind harder. Rise higher. ✨" });

    const row = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId("menu_home").setLabel("🏠 Back").setStyle(ButtonStyle.Danger),
    );
    return i.update({ embeds: [embed], components: [row] });
  }

  // ── Notes Menu ──
  if (id === "menu_notes") {
    const embed = new EmbedBuilder()
      .setTitle("📝 Your Second Brain")
      .setDescription("Save anything you learn. Search it later. Never forget again.")
      .setColor(0x3b82f6);

    const row = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId("note_add").setLabel("➕ Add Note").setStyle(ButtonStyle.Success),
      new ButtonBuilder().setCustomId("note_search").setLabel("🔍 Search Notes").setStyle(ButtonStyle.Primary),
      new ButtonBuilder().setCustomId("note_recent").setLabel("📋 Recent Notes").setStyle(ButtonStyle.Secondary),
      new ButtonBuilder().setCustomId("menu_home").setLabel("🏠 Back").setStyle(ButtonStyle.Danger),
    );
    return i.update({ embeds: [embed], components: [row] });
  }

  // ── Add Note Modal ──
  if (id === "note_add") {
    const modal = new ModalBuilder().setCustomId("modal_note_add").setTitle("📝 Add a Note");
    modal.addComponents(
      new ActionRowBuilder().addComponents(
        new TextInputBuilder().setCustomId("note_title").setLabel("Title").setStyle(TextInputStyle.Short).setRequired(true)
      ),
      new ActionRowBuilder().addComponents(
        new TextInputBuilder().setCustomId("note_content").setLabel("Content").setStyle(TextInputStyle.Paragraph).setRequired(true)
      ),
      new ActionRowBuilder().addComponents(
        new TextInputBuilder().setCustomId("note_tags").setLabel("Tags (comma separated)").setStyle(TextInputStyle.Short).setRequired(false)
      ),
    );
    return i.showModal(modal);
  }

  // ── Search Note Modal ──
  if (id === "note_search") {
    const modal = new ModalBuilder().setCustomId("modal_note_search").setTitle("🔍 Search Notes");
    modal.addComponents(
      new ActionRowBuilder().addComponents(
        new TextInputBuilder().setCustomId("search_query").setLabel("Search by title or tag").setStyle(TextInputStyle.Short).setRequired(true)
      ),
    );
    return i.showModal(modal);
  }

  // ── Recent Notes ──
  if (id === "note_recent") {
    await ensureUser(i.user.id);
    const notes = await prisma.note.findMany({
      where: { userId: i.user.id },
      take: 5,
      orderBy: { createdAt: "desc" },
    });

    const desc = notes.length
      ? notes.map((n) => `**#${n.id} — ${n.title}**\n${n.content.slice(0, 80)}${n.content.length > 80 ? "..." : ""}\nTags: ${n.tags.map((t) => `\`${t}\``).join(" ") || "None"}`).join("\n\n")
      : "No notes yet! Start saving what you learn 🧠";

    const embed = new EmbedBuilder()
      .setTitle("📋 Recent Notes")
      .setDescription(desc)
      .setColor(0x3b82f6);

    const row = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId("note_add").setLabel("➕ Add Note").setStyle(ButtonStyle.Success),
      new ButtonBuilder().setCustomId("menu_notes").setLabel("⬅️ Back").setStyle(ButtonStyle.Secondary),
      new ButtonBuilder().setCustomId("menu_home").setLabel("🏠 Home").setStyle(ButtonStyle.Danger),
    );
    return i.update({ embeds: [embed], components: [row] });
  }

  // ── Ask Meera Modal ──
  if (id === "menu_ask") {
    const modal = new ModalBuilder().setCustomId("modal_ask").setTitle("🧠 Ask Meera Anything");
    modal.addComponents(
      new ActionRowBuilder().addComponents(
        new TextInputBuilder().setCustomId("ask_question").setLabel("What's on your mind?").setStyle(TextInputStyle.Paragraph).setRequired(true).setPlaceholder("e.g. Explain binary search with an example...")
      ),
    );
    return i.showModal(modal);
  }

  // ── Hint ──
  if (id.startsWith("hint_")) {
    const qId = parseInt(id.split("_")[1]);
    const q = await prisma.question.findUnique({ where: { id: qId } });
    return i.reply({ content: `💡 **Hint:** ||${q?.hints || "No hint available"}||`, ephemeral: true });
  }

  // ── Voice Join ──
  if (id === "menu_voice_join") {
    const vc = i.member.voice.channel;
    if (!vc) {
      return i.reply({ content: "Pehle kisi VC mein ja, phir click kar! 🎙️", ephemeral: true });
    }
    if (isInVC(i.guild.id)) {
      leaveVC(i.guild.id);
      return i.reply({ content: "👋 Left the VC. See you later!", ephemeral: true });
    }
    joinVC(vc);
    return i.reply({ content: `🎙️ Joined **${vc.name}**! Say **"Hey Meera"** to start talking.`, ephemeral: false });
  }

  // ── Submit Solution Modal ──
  if (id.startsWith("submit_")) {
    const qId = id.split("_")[1];
    const modal = new ModalBuilder().setCustomId(`modal_submit_${qId}`).setTitle("✍️ Submit Your Solution");
    modal.addComponents(
      new ActionRowBuilder().addComponents(
        new TextInputBuilder().setCustomId("sub_lang").setLabel("Language (python/javascript/cpp/java)").setStyle(TextInputStyle.Short).setRequired(true).setPlaceholder("python")
      ),
      new ActionRowBuilder().addComponents(
        new TextInputBuilder().setCustomId("sub_code").setLabel("Your Code").setStyle(TextInputStyle.Paragraph).setRequired(true)
      ),
    );
    return i.showModal(modal);
  }
}

async function handleSelect(i) {
  if (i.customId === "select_topic") {
    const topic = i.values[0];
    const questions = await prisma.question.findMany({ where: { topic } });

    if (!questions.length) {
      return i.update({ content: "No questions for this topic yet!", embeds: [], components: [] });
    }

    const q = questions[Math.floor(Math.random() * questions.length)];
    const embed = buildQuestionEmbed(q, `📚 ${topic}`);

    const row = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId(`hint_${q.id}`).setLabel("💡 Show Hint").setStyle(ButtonStyle.Secondary),
      new ButtonBuilder().setCustomId(`submit_${q.id}`).setLabel("✍️ Submit Solution").setStyle(ButtonStyle.Success),
      new ButtonBuilder().setCustomId("menu_topics").setLabel("🔄 Another Topic").setStyle(ButtonStyle.Primary),
      new ButtonBuilder().setCustomId("menu_home").setLabel("🏠 Home").setStyle(ButtonStyle.Danger),
    );
    return i.update({ embeds: [embed], components: [row] });
  }
}

async function handleModal(i) {
  // ── Note Add ──
  if (i.customId === "modal_note_add") {
    await ensureUser(i.user.id);
    const title = i.fields.getTextInputValue("note_title");
    const content = i.fields.getTextInputValue("note_content");
    const tagsRaw = i.fields.getTextInputValue("note_tags");
    const tags = tagsRaw ? tagsRaw.split(",").map((t) => t.trim().toLowerCase()).filter(Boolean) : [];

    const note = await prisma.note.create({
      data: { userId: i.user.id, title, content, tags },
    });

    const embed = new EmbedBuilder()
      .setTitle("📝 Note Saved!")
      .setDescription(`Got it! I'll remember this for you.`)
      .addFields(
        { name: "Title", value: title },
        { name: "Tags", value: tags.length ? tags.map((t) => `\`${t}\``).join(" ") : "None" },
      )
      .setColor(0x22c55e)
      .setFooter({ text: `Note #${note.id}` });

    const row = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId("note_add").setLabel("➕ Add Another").setStyle(ButtonStyle.Success),
      new ButtonBuilder().setCustomId("menu_notes").setLabel("⬅️ Notes").setStyle(ButtonStyle.Secondary),
      new ButtonBuilder().setCustomId("menu_home").setLabel("🏠 Home").setStyle(ButtonStyle.Danger),
    );
    return i.reply({ embeds: [embed], components: [row] });
  }

  // ── Note Search ──
  if (i.customId === "modal_note_search") {
    await ensureUser(i.user.id);
    const query = i.fields.getTextInputValue("search_query").toLowerCase();

    const notes = await prisma.note.findMany({
      where: {
        userId: i.user.id,
        OR: [
          { title: { contains: query, mode: "insensitive" } },
          { tags: { has: query } },
        ],
      },
      take: 5,
      orderBy: { createdAt: "desc" },
    });

    const desc = notes.length
      ? notes.map((n) => `**#${n.id} — ${n.title}**\n${n.content.slice(0, 80)}${n.content.length > 80 ? "..." : ""}\nTags: ${n.tags.map((t) => `\`${t}\``).join(" ") || "None"}`).join("\n\n")
      : "Nothing found! Try a different keyword 🤔";

    const embed = new EmbedBuilder()
      .setTitle(`🔍 Results for "${query}"`)
      .setDescription(desc)
      .setColor(0x3b82f6)
      .setFooter({ text: `${notes.length} result(s)` });

    const row = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId("note_search").setLabel("🔍 Search Again").setStyle(ButtonStyle.Primary),
      new ButtonBuilder().setCustomId("menu_notes").setLabel("⬅️ Notes").setStyle(ButtonStyle.Secondary),
      new ButtonBuilder().setCustomId("menu_home").setLabel("🏠 Home").setStyle(ButtonStyle.Danger),
    );
    return i.reply({ embeds: [embed], components: [row] });
  }

  // ── Ask Meera ──
  if (i.customId === "modal_ask") {
    await i.deferReply();
    try {
      const question = i.fields.getTextInputValue("ask_question");
      const answer = await chat(question);
      const chunks = answer.match(/[\s\S]{1,4000}/g) || [answer];

      const embed = new EmbedBuilder()
        .setTitle("✨ Meera says...")
        .setDescription(chunks[0])
        .setColor(0x7c3aed)
        .setFooter({ text: "Powered by Meera AI 💜" });

      const row = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId("menu_ask").setLabel("🧠 Ask Again").setStyle(ButtonStyle.Primary),
        new ButtonBuilder().setCustomId("menu_home").setLabel("🏠 Home").setStyle(ButtonStyle.Danger),
      );

      await i.editReply({ embeds: [embed], components: [row] });

      for (let idx = 1; idx < chunks.length; idx++) {
        await i.followUp({
          embeds: [new EmbedBuilder().setDescription(chunks[idx]).setColor(0x7c3aed)],
        });
      }
    } catch (err) {
      console.error("AI error:", err);
      await i.editReply({ content: "My brain glitched for a sec 😅 Try again?" });
    }
  }

  // ── Submit Solution ──
  if (i.customId.startsWith("modal_submit_")) {
    const qId = parseInt(i.customId.split("_")[2]);
    const language = i.fields.getTextInputValue("sub_lang").toLowerCase().trim();
    const code = i.fields.getTextInputValue("sub_code");

    await ensureUser(i.user.id);
    const question = await prisma.question.findUnique({ where: { id: qId } });
    if (!question) return i.reply({ content: "Question not found!", ephemeral: true });

    await prisma.submission.create({
      data: { userId: i.user.id, questionId: qId, code, language },
    });

    const { xpGained, totalXP, streak } = await awardXP(i.user.id, question.difficulty);

    const embed = new EmbedBuilder()
      .setTitle("✅ Solution Submitted!")
      .setDescription(`Nice work on **${question.title}**! Keep that momentum going 🔥`)
      .addFields(
        { name: "⚡ XP Earned", value: `+${xpGained}`, inline: true },
        { name: "✨ Total XP", value: `${totalXP}`, inline: true },
        { name: "🔥 Streak", value: `${streak} day(s)`, inline: true },
      )
      .setColor(0x22c55e);

    const row = new ActionRowBuilder().addComponents(
      new ButtonBuilder().setCustomId("menu_daily").setLabel("📅 Next Question").setStyle(ButtonStyle.Primary),
      new ButtonBuilder().setCustomId("menu_progress").setLabel("📊 My Progress").setStyle(ButtonStyle.Success),
      new ButtonBuilder().setCustomId("menu_home").setLabel("🏠 Home").setStyle(ButtonStyle.Danger),
    );
    return i.reply({ embeds: [embed], components: [row] });
  }
}

function buildQuestionEmbed(q, label) {
  const colors = { Easy: 0x22c55e, Medium: 0xf59e0b, Hard: 0xef4444 };
  return new EmbedBuilder()
    .setTitle(`${label} — ${q.title}`)
    .setDescription(q.description)
    .addFields(
      { name: "Difficulty", value: q.difficulty, inline: true },
      { name: "Topic", value: q.topic, inline: true },
      { name: "Example", value: `\`\`\`\n${q.examples}\n\`\`\`` },
    )
    .setColor(colors[q.difficulty] || 0x7c3aed)
    .setFooter({ text: `Question #${q.id}` });
}
