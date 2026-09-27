const assert = require('node:assert/strict');
const { test } = require('node:test');
const { invalidCommits } = require('./release-commits.cjs');

const commit = (message, parents = [{}]) => ({ commit: { message }, parents });

test('accepts categories, scopes, breaking markers, release commits and real merges', () => {
  const commits = [
    commit('feat: add search'), commit('fix(cli)!: change output\n\nBREAKING CHANGE: output changed'),
    commit('chore(main): release 0.2.0'), commit('Merge branch main', [{}, {}]),
  ];
  assert.deepEqual(invalidCommits(commits), []);
});

test('rejects missing or malformed categories and fake merges', () => {
  const commits = ['Add search', 'fix: ', 'feat(): invalid scope', 'unknown: new feature',
    'Merge branch main', 'fix:\nnot a subject'].map(message => commit(message));
  assert.deepEqual(invalidCommits(commits), commits);
});
