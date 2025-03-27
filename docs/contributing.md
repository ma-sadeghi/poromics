# Contributing to **Poromics**

Poromics is a collection of tools to rapidly estimate transport properties of 2D/3D images of porous materials. We use Poromics regularly in our research, and we hope you will too. The package is open-source, and we welcome contributions from the community. This document describes how to get involved.

Before you start you'll need to set up a free GitHub account and sign in. Here are some [instructions][link_signupinstructions] to get started.

## Ways to Contribute

### Open a New Issue

We use Github to track [issues][link_issues]. Issues can take the form of:

(a) bug reports such as a function producing an error or odd result in some circumstances.

(b) feature requests such a suggesting a new function be added to the package, presumably based on some literature report that describes it, or enhancements to an existing function.

(c) general usage questions where the documentation is not clear and you need help getting a function to work as desired. This is actually a bug report in disguise since it means there is a problem with the documentation.

### Addressing Open Issues

Help fixing open [issues][link_issues] is always welcome; however, the learning curve for submitting new code to any repo on Github is a bit intimidating. The process is as follows:

a) [Fork][link_fork] Poromics to your own Github account. This lets you work on the code since you are the owner of that forked copy.

b) Pull the code to your local machine using some Git client. We suggest [GitKraken][link_gitkraken]. For help using the Git version control system, see [these resources][link_using_git].

c) Create a new branch, with a useful name like "fix_issue_41" or "add_awesome_solver", then checkout that branch.

d) Edit the code as desired, either fixing or adding something. You'll need to know Python and the various packages in the [SciPy][link_scipy] stack for this part.

e) Push the changes back to Github, to your own repo.

f) Navigate to the [pull requests area][link_pull_requests] on the Poromics repo, then click the "new pull request" button. As the name suggests, you are [requesting us to pull][link_pullrequest] your code in to our repo. You'll want to select the correct branch on your repo (e.g. "add_awesome_filter") and the "dev" branch on Poromics.

g) This will trigger several things on our repo, including most importantly a conversation between you and the Poromics team about your code. After any fine-tuning is done, we will merge your code into Poromics, and your contribution will be immortalized in Poromics.

[link_issues]: https://github.com/ma-sadeghi/poromics/issues
[link_gitkraken]: https://www.gitkraken.com/
[link_pull_requests]: https://github.com/ma-sadeghi/poromics/pulls
[link_fork]: https://help.github.com/articles/fork-a-repo/
[link_signupinstructions]: https://help.github.com/articles/signing-up-for-a-new-github-account
[link_pullrequest]: https://help.github.com/articles/creating-a-pull-request/
[link_using_git]: http://try.github.io/
[link_scipy]: https://www.scipy.org/

!!! note

    Adapted from PoreSpy's [contributing guide](https://github.com/PMEAL/porespy/blob/dev/CONTRIBUTING.md).
