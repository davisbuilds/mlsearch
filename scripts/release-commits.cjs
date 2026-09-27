// Full commit history is preserved, so each non-merge commit needs a category.
const conventional = /^(feat|fix|perf|docs|test|chore|build|ci|style|refactor|revert)(\([^\r\n()]+\))?!?: \S.*$/;

function invalidCommits(commits) {
  return commits.filter(({ commit, parents }) => {
    if (parents.length > 1) return false;
    const subject = commit.message.split(/\r?\n/, 1)[0];
    return !conventional.test(subject);
  });
}

async function commitsForEvent(context, github) {
  if (context.eventName === 'pull_request') {
    if (context.payload.pull_request.commits > 250) {
      throw new Error('Commit category check supports up to 250 PR commits; split this PR.');
    }
    return github.paginate(github.rest.pulls.listCommits, {
      ...context.repo, pull_number: context.issue.number, per_page: 100
    });
  }
  if (context.eventName !== 'push') throw new Error('Unsupported commit event.');
  const { before, after, deleted } = context.payload;
  if (deleted || !/^[0-9a-f]{40}$/.test(before) || !/^[0-9a-f]{40}$/.test(after) || /^0+$/.test(before)) {
    throw new Error('Push needs an existing base and head; preserve main history.');
  }
  const commits = [];
  let total;
  for (let page = 1; page <= 3; page++) {
    const { data } = await github.rest.repos.compareCommitsWithBasehead({
      ...context.repo, basehead: `${before}...${after}`, per_page: 100, page
    });
    total = data.total_commits;
    if (!Number.isInteger(total) || total > 250 || data.status === 'diverged' || data.status === 'behind') {
      throw new Error('Push must preserve history and contain at most 250 commits; split the push.');
    }
    commits.push(...data.commits);
    if (commits.length === total) return commits;
    if (!data.commits.length) break;
  }
  throw new Error('Could not collect the full pushed commit range.');
}

module.exports = { invalidCommits, commitsForEvent };
