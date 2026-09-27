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

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { commitsForEvent } = require('./release-commits.cjs');

function history(t) {
  const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'mlsearch-history-'));
  t.after(() => fs.rmSync(cwd, {recursive:true,force:true}));
  const git = (...args) => execFileSync('git', args, {cwd, encoding:'utf8'}).trim();
  git('init','-q');
  git('config','user.name','Release Test');
  git('config','user.email','release@example.com');
  const add = message => {
    git('-c','commit.gpgsign=false','commit','-q','--allow-empty','-m',message);
    return git('rev-parse','HEAD');
  };
  const bootstrap = add('chore: bootstrap');
  fs.writeFileSync(path.join(cwd,'release-please-config.json'),JSON.stringify({'bootstrap-sha':bootstrap}));
  fs.writeFileSync(path.join(cwd,'.release-please-manifest.json'),JSON.stringify({'.':'0.1.0'}));
  const push = (before,after) => ({eventName:'push',repo:{owner:'owner',repo:'repo'},payload:{before,after}});
  const github = {rest:{repos:{compareCommitsWithBasehead:async ({basehead,per_page,page}) => {
    const [base,head] = basehead.split('...');
    let status = 'ahead';
    try { git('merge-base','--is-ancestor',base,head); } catch { status = 'diverged'; }
    const shas = git('rev-list','--reverse',`${base}..${head}`).split('\n').filter(Boolean);
    const commits = shas.map(sha => ({sha,
      commit:{message:git('show','-s','--format=%s',sha)},
      parents:git('show','-s','--format=%P',sha).split(' ').filter(Boolean).map(sha=>({sha}))}));
    return {data:{status,total_commits:commits.length,commits:commits.slice((page-1)*per_page,page*per_page)}};
  }}}};
  return {cwd,git,add,bootstrap,push,github};
}

test('later valid push cannot forget an earlier unclassified main commit', async t => {
  const h = history(t);
  const bad = h.add('Add unclassified consumer change');
  const good = h.add('fix: later valid change');
  const commits = await commitsForEvent(h.push(bad,good), h.github, h.cwd);
  assert.deepEqual(invalidCommits(commits).map(c=>c.commit.message), ['Add unclassified consumer change']);
  assert.equal(commits.length,2);
});

test('real manifest tag bounds released history; missing new tag falls back', async t => {
  const h = history(t);
  const released = h.add('Old released subject');
  h.git('-c','tag.gpgsign=false','tag','-a','v0.1.0','-m','Actual release',released);
  const good = h.add('fix: new change');
  assert.deepEqual(invalidCommits(await commitsForEvent(h.push(released,good),h.github,h.cwd)), []);
  fs.writeFileSync(path.join(h.cwd,'.release-please-manifest.json'),JSON.stringify({'.':'0.2.0'}));
  assert.equal(invalidCommits(await commitsForEvent(h.push(released,good),h.github,h.cwd)).length,1);
});

test('rejects rewritten, unavailable, and zero-base push history', async t => {
  const h = history(t);
  const before = h.add('fix: first branch');
  h.git('checkout','-q','--detach',h.bootstrap);
  const after = h.add('fix: rewritten branch');
  for (const base of [before,'f'.repeat(40),'0'.repeat(40)]) {
    await assert.rejects(commitsForEvent(h.push(base,after),h.github,h.cwd));
  }
});

test('accumulated release history is not capped at 250 commits', async t => {
  const h = history(t);
  let before = h.bootstrap;
  for (let i=0;i<251;i++) before = h.add('chore: maintenance');
  const after = h.add('fix: latest change');
  assert.equal((await commitsForEvent(h.push(before,after),h.github,h.cwd)).length,252);
});


test('malformed manifest SemVer rejects instead of silently using bootstrap', async t => {
  const h = history(t);
  const after = h.add('fix: valid consumer change');
  const invalid = ['01.2.3','1.02.3','1.2.03','1.2.3-01','1.2.3-alpha..beta',
    '1.2.3+build..id','1.2.3-','1.2.3+','1.2.3\n','1.2.3-α','v1.2.3',null,123];
  for (const version of invalid) {
    fs.writeFileSync(path.join(h.cwd,'.release-please-manifest.json'),JSON.stringify({'.':version}));
    await assert.rejects(commitsForEvent(h.push(h.bootstrap,after),{},h.cwd),
      /Invalid release manifest version/, `reject ${JSON.stringify(version)}`);
  }
});

test('valid SemVer retains bootstrap fallback and exact real-tag selection', async t => {
  const h = history(t);
  const released = h.add('Old released subject');
  const after = h.add('fix: valid consumer change');
  const versions = ['0.0.0','1.2.3','1.2.3-0','1.2.3-alpha.1','1.2.3-0alpha',
    '1.2.3+01.build','1.2.3-alpha-1+build.01','1.2.3--'];
  for (const version of versions) {
    fs.writeFileSync(path.join(h.cwd,'.release-please-manifest.json'),JSON.stringify({'.':version}));
    assert.equal(invalidCommits(await commitsForEvent(h.push(released,after),{},h.cwd)).length,1);
    h.git('-c','tag.gpgsign=false','tag',`v${version}`,released);
    assert.deepEqual(invalidCommits(await commitsForEvent(h.push(released,after),{},h.cwd)),[]);
  }
});
