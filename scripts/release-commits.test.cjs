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

test('collects main push commits across compare pages', async () => {
  const { commitsForEvent } = require('./release-commits.cjs');
  const context = { eventName: 'push', repo: {owner: 'owner', repo: 'repo'},
    payload: {before: 'a'.repeat(40), after: 'b'.repeat(40)} };
  const pages = [[commit('feat: consumer feature')], [commit('Add unclassified change')]];
  const github = { rest: {repos: { compareCommitsWithBasehead: async ({page}) =>
    ({data: {total_commits: 2, commits: pages[page - 1]}}) }} };
  const commits = await commitsForEvent(context, github);
  assert.equal(commits.length, 2);
  assert.equal(invalidCommits(commits).length, 1);
});

test('fails closed for oversized or incomplete pushed history', async () => {
  const { commitsForEvent } = require('./release-commits.cjs');
  const context = { eventName: 'push', repo: {owner: 'owner', repo: 'repo'},
    payload: {before: 'a'.repeat(40), after: 'b'.repeat(40)} };
  for (const data of [{total_commits: 251, commits: []}, {total_commits: 1, commits: []}]) {
    const github = { rest: {repos: {compareCommitsWithBasehead: async () => ({data})}} };
    await assert.rejects(commitsForEvent(context, github));
  }
  await assert.rejects(commitsForEvent({...context, payload: {before:'0'.repeat(40), after:'b'.repeat(40)}}, {}));
});
