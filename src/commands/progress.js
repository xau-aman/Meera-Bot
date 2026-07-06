const { SlashCommandBuilder, EmbedBuilder } = require("discord.js");
const prisma = require("../lib/db");
const { ensureUser } = require("../lib/xp");

module.exports = {
  data: new SlashCommandBuilder()
    .setName("progress")
    .setDescription("View your coding progress and stats"),

  async execute(interaction) {
    const user = await ensureUser(interaction.user.id);
    const submissions = await prisma.submission.count({ where: { userId: interaction.user.id } });

    const byDifficulty = await prisma.submission.groupBy({
      by: ["questionId"],
      where: { userId: interaction.user.id },
    });

    const level = Math.floor(user.xp / 100) + 1;
    const xpToNext = 100 - (user.xp % 100);
    const bar = "█".repeat(Math.floor((user.xp % 100) / 10)) + "░".repeat(10 - Math.floor((user.xp % 100) / 10));

    const embed = new EmbedBuilder()
      .setTitle(`📊 ${interaction.user.username}'s Progress`)
      .addFields(
        { name: "Level", value: `${level}`, inline: true },
        { name: "Total XP", value: `${user.xp}`, inline: true },
        { name: "🔥 Streak", value: `${user.streak} day(s)`, inline: true },
        { name: "Problems Solved", value: `${submissions}`, inline: true },
        { name: "Progress to Next Level", value: `${bar} (${xpToNext} XP to go)` }
      )
      .setColor(0x7c3aed)
      .setFooter({ text: "Meera — Keep pushing, you're doing great! ✨" });

    return interaction.reply({ embeds: [embed] });
  },
};
