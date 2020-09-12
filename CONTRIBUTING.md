# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Get Started!
Ready to contribute? Here's how to set up `pugh_torch` for local development.

1. Fork the `pugh_torch` repo on GitHub.

2. Clone your fork locally:

    ```bash
    git clone git@github.com:{your_name_here}/pugh_torch.git
    ```

3. Install the project in editable mode. (It is also recommended to work in a virtualenv or anaconda environment):

    ```bash
    cd pugh_torch/
    pip install -e .[dev]
    ```

4. Create a branch for local development:

    ```bash
    git checkout -b {your_development_type}/short-description
    ```

    Ex: feature/read-tiff-files or bugfix/handle-file-not-found<br>
    Now you can make your changes locally.

5. When you're done making changes, check that your changes pass linting and
   tests, including testing other Python versions with make:

    ```bash
    make build
    ```

6. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Resolves gh-###. Your detailed description of your changes."
    git push origin {your_development_type}/short-description
    ```

7. Submit a pull request through the GitHub website.

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run:

```bash
$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags
git branch -D stable
git checkout -b stable
git push --set-upstream origin stable -f
```

This will release a new package version on Git + GitHub and publish to PyPI.


## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## The Four Commands You Need To Know
1. `pip install -e .[dev]`

    This will install your package in editable mode with all the required development
    dependencies (i.e. `tox`).

2. `make build`

    This will run `tox` which will run all your tests in both Python 3.7
    and Python 3.8 as well as linting your code.

3. `make clean`

    This will clean up various Python and build generated files so that you can ensure
    that you are working in a clean environment.

4. `make docs`

    This will generate and launch a web browser to view the most up-to-date
    documentation for your Python package.

#### Additional Optional Setup Steps:
* Turn your project into a GitHub repository:
  * Make sure you have `git` installed, if you don't, [follow these instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
  * Make an account on [github.com](https://github.com)
  * Go to [make a new repository](https://github.com/new)
  * _Recommendations:_
    * _It is strongly recommended to make the repository name the same as the Python
    package name_
    * _A lot of the following optional steps are *free* if the repository is Public,
    plus open source is cool_
  * After a GitHub repo has been created, run the following commands:
    * `git remote add origin git@github.com:BrianPugh/pugh_torch.git`
    * `git push -u origin master`
* Register pugh_torch with Codecov:
  * Make an account on [codecov.io](https://codecov.io)
  (Recommended to sign in with GitHub)
  * Select `BrianPugh` and click: `Add new repository`
  * Copy the token provided, go to your [GitHub repository's settings and under the `Secrets` tab](https://github.com/BrianPugh/pugh_torch/settings/secrets),
  add a secret called `CODECOV_TOKEN` with the token you just copied.
  Don't worry, no one will see this token because it will be encrypted.
* Generate and add an access token as a secret to the repository for auto documentation
generation to work
  * Go to your [GitHub account's Personal Access Tokens page](https://github.com/settings/tokens)
  * Click: `Generate new token`
  * _Recommendations:_
    * _Name the token: "Auto-Documentation Generation" or similar so you know what it
    is being used for later_
    * _Select only: `repo:status`, `repo_deployment`, and `public_repo` to limit what
    this token has access to_
  * Copy the newly generated token
  * Go to your [GitHub repository's settings and under the `Secrets` tab](https://github.com/BrianPugh/pugh_torch/settings/secrets),
  add a secret called `ACCESS_TOKEN` with the personal access token you just created.
  Don't worry, no one will see this password because it will be encrypted.
* Register your project with PyPI:
  * Make an account on [pypi.org](https://pypi.org)
  * Go to your [GitHub repository's settings and under the `Secrets` tab](https://github.com/BrianPugh/pugh_torch/settings/secrets),
  add a secret called `PYPI_TOKEN` with your password for your PyPI account.
  Don't worry, no one will see this password because it will be encrypted.
  * Next time you push to the branch: `stable`, GitHub actions will build and deploy
  your Python package to PyPI.
  * _Recommendation: Prior to pushing to `stable` it is recommended to install and run
  `bumpversion` as this will,
  tag a git commit for release and update the `setup.py` version number._
* Add branch protections to `master` and `stable`
    * To protect from just anyone pushing to `master` or `stable` (the branches with
    more tests and deploy
    configurations)
    * Go to your [GitHub repository's settings and under the `Branches` tab](https://github.com/BrianPugh/pugh_torch/settings/branches), click `Add rule` and select the
    settings you believe best.
    * _Recommendations:_
      * _Require pull request reviews before merging_
      * _Require status checks to pass before merging (Recommended: lint and test)_

#### Suggested Git Branch Strategy
1. `master` is for the most up-to-date development, very rarely should you directly
commit to this branch. GitHub Actions will run on every push and on a CRON to this
branch but still recommended to commit to your development branches and make pull
requests to master.
2. `stable` is for releases only. When you want to release your project on PyPI, simply
make a PR from `master` to `stable`, this template will handle the rest as long as you
have added your PyPI information described in the above **Optional Steps** section.
3. Your day-to-day work should exist on branches separate from `master`. Even if it is
just yourself working on the repository, make a PR from your working branch to `master`
so that you can ensure your commits don't break the development head. GitHub Actions
will run on every push to any branch or any pull request from any branch to any other
branch.
4. It is recommended to use "Squash and Merge" commits when committing PR's. It makes
each set of changes to `master` atomic and as a side effect naturally encourages small
well defined PR's.
5. GitHub's UI is bad for rebasing `master` onto `stable`, as it simply adds the
commits to the other branch instead of properly rebasing from what I can tell. You
should always rebase locally on the CLI until they fix it.



