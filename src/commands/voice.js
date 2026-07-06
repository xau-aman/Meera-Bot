const { SlashCommandBuilder } = require("discord.js");
const { joinVC, leaveVC, isInVC } = require("../voice/voiceHandler");

module.exports = {
  data: new SlashCommandBuilder()
    .setName("voice")
    .setDescription("Meera voice channel controls")
    .addSubcommand((sub) =>
      sub.setName("join").setDescription("Meera joins your voice channel")
    )
    .addSubcommand((sub) =>
      sub.setName("leave").setDescription("Meera leaves the voice channel")
    ),

  async execute(interaction) {
    const sub = interaction.options.getSubcommand();

    if (sub === "join") {
      const vc = interaction.member.voice.channel;
      if (!vc) {
        return interaction.reply({ content: "Pehle kisi VC mein ja! 🎙️", ephemeral: true });
      }

      if (isInVC(interaction.guild.id)) {
        return interaction.reply({ content: "Main already VC mein hoon! Say \"Hey Meera\" to start talking 💜", ephemeral: true });
      }

      joinVC(vc);
      return interaction.reply(`🎙️ Joined **${vc.name}**! Say **"Hey Meera"** to start a conversation.`);
    }

    if (sub === "leave") {
      if (!isInVC(interaction.guild.id)) {
        return interaction.reply({ content: "Main kisi VC mein nahi hoon 🤷‍♀️", ephemeral: true });
      }

      leaveVC(interaction.guild.id);
      return interaction.reply("👋 Left the voice channel. Talk to you later!");
    }
  },
};
