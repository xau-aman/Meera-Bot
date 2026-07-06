const { EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle } = require("discord.js");

module.exports = {
  name: "messageCreate",
  async execute(message) {
    if (message.author.bot) return;
    if (!message.mentions.has(message.client.user)) return;

    const embed = new EmbedBuilder()
      .setTitle("✨ Hey, I'm Meera!")
      .setDescription(
        `Hey **${message.author.displayName}**! I'm your personal AI mentor for coding, careers & interviews.\n\nWhat do you wanna do today? Pick something below 👇`
      )
      .setColor(0x7c3aed)
      .setThumbnail(message.client.user.displayAvatarURL())
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

    await message.reply({ embeds: [embed], components: [row1, row2] });
  },
};
