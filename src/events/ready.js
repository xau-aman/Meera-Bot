module.exports = {
  name: "ready",
  once: true,
  execute(client) {
    console.log(`Aurora is online as ${client.user.tag}`);
    client.user.setActivity("your mentor | @mention me", { type: 3 });
  },
};
