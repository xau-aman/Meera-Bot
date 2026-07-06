const { SlashCommandBuilder, EmbedBuilder } = require("discord.js");
const { chat } = require("../lib/ai");

module.exports = {
  data: new SlashCommandBuilder()
    .setName("ask")
    .setDescription("Ask Meera anything about coding, DSA, or CS concepts")
    .addStringOption((o) =>
      o.setName("question").setDescription("Your question").setRequired(true)
    ),

  async execute(interaction) {
    await interaction.deferReply();

    try {
      const question = interaction.options.getString("question");
      const answer = await chat(question);

      // Discord embeds have a 4096 char limit for description
      const chunks = answer.match(/[\s\S]{1,4000}/g) || [answer];

      const embed = new EmbedBuilder()
        .setTitle("✨ Meera says...")
        .setDescription(chunks[0])
        .setColor(0x7c3aed)
        .setFooter({ text: "Powered by Meera AI • Ask me anything!" });

      await interaction.editReply({ embeds: [embed] });

      // Send overflow as follow-ups
      for (let i = 1; i < chunks.length; i++) {
        await interaction.followUp({
          embeds: [new EmbedBuilder().setDescription(chunks[i]).setColor(0x7c3aed)],
        });
      }
    } catch (err) {
      console.error("AI error:", err);
      await interaction.editReply({ content: "Hmm, my brain glitched. Try again? 🤖" });
    }
  },
};
