const fs = require('node:fs');
const path = require('node:path');
const { execFileSync, spawnSync } = require('node:child_process');

// SemVer 2.0.0 grammar: https://semver.org/#backusnaur-form-grammar-for-valid-semver-versions
const numeric = '(?:0|[1-9][0-9]*)';
const prerelease = `(?:${numeric}|[0-9]*[A-Za-z-][0-9A-Za-z-]*)`;
const build = '[0-9A-Za-z-]+';
const semver = new RegExp(
  String.raw`^${numeric}\.${numeric}\.${numeric}` +
  String.raw`(?:-${prerelease}(?:\.${prerelease})*)?` +
  String.raw`(?:\+${build}(?:\.${build})*)?(?![\s\S])`
);

// Full commit history is preserved, so each non-merge commit needs a category.
const conventional = /^(feat|fix|perf|docs|test|chore|build|ci|style|refactor|revert)(\([^\r\n()]+\))?!?: \S.*$/;

function invalidCommits(commits) {
  return commits.filter(({ commit, parents }) => {
    if (parents.length > 1) return false;
    const subject = commit.message.split(/\r?\n/, 1)[0];
    return !conventional.test(subject);
  });
}

async function commitsForEvent(context, github, cwd = process.cwd()) {
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
  const git = (...args) => execFileSync('git', args, {
    cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 64 * 1024 * 1024
  }).trim();
  git('merge-base', '--is-ancestor', before, after);
  const manifest = JSON.parse(fs.readFileSync(path.join(cwd, '.release-please-manifest.json')));
  const configuration = JSON.parse(fs.readFileSync(path.join(cwd, 'release-please-config.json')));
  const version = manifest['.'];
  if (typeof version !== 'string' || !semver.test(version)) {
    throw new Error('Invalid release manifest version.');
  }
  const tag = `refs/tags/v${version}`;
  const exists = spawnSync('git', ['show-ref', '--verify', '--quiet', tag], {cwd});
  if (exists.error || ![0,1].includes(exists.status)) throw new Error('Could not inspect release tag.');
  const baseline = exists.status === 0
    ? git('rev-parse', '--verify', `${tag}^{commit}`)
    : configuration['bootstrap-sha'];
  if (!/^[0-9a-f]{40}$/.test(baseline) || /^0+$/.test(baseline)) {
    throw new Error('A real release tag or configured bootstrap commit is required.');
  }
  git('merge-base', '--is-ancestor', baseline, after);
  // Check the entire unreleased window, including earlier failed main pushes.
  const records = git('log', '--format=%H%x00%P%x00%s%x00', `${baseline}..${after}`);
  if (!records) return [];
  const fields = records.split('\0');
  if (fields.pop() !== '' || fields.length % 3) throw new Error('Incomplete release history.');
  const commits = [];
  for (let i = 0; i < fields.length; i += 3) {
    commits.push({sha:fields[i].trim(), parents:fields[i+1].split(' ').filter(Boolean),
      commit:{message:fields[i+2]}});
  }
  return commits;
}

module.exports = { invalidCommits, commitsForEvent };
