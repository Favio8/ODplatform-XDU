const ANSI_PATTERN =
  // Matches common ANSI escape/control sequences emitted by CLI tools.
  // eslint-disable-next-line no-control-regex
  /\u001B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g;

export function stripAnsi(value: string | null | undefined): string {
  if (!value) return "";
  return value.replace(ANSI_PATTERN, "").replace(/\r/g, "").trim();
}
