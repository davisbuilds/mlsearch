// Full commit history is preserved, so each non-merge commit needs a category.
const conventional = /^(feat|fix|perf|docs|test|chore|build|ci|style|refactor|revert)(\([^\r\n()]+\))?!?: \S.*$/;

function invalidCommits(commits) {
  return commits.filter(({ commit, parents }) => {
    if (parents.length > 1) return false;
    const subject = commit.message.split(/\r?\n/, 1)[0];
    return !conventional.test(subject);
  });
}

module.exports = { invalidCommits };
