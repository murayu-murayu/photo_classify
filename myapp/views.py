from django.shortcuts import render, resolve_url, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.views.generic import (
    TemplateView,
    CreateView,
    DetailView,
    UpdateView,
    DeleteView,
    ListView,
)
from .models import Post, Like, Category
from django.urls import reverse_lazy
from .forms import PostForm, LoginForm, SignUpForm, SearchForm
from django.contrib import messages
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth import login
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.core.paginator import Paginator

from .models import Photo
from .models import CustomUser



class OnlyMyPostMixin(UserPassesTestMixin):
    raise_exceptions = True

    def test_func(self):
        post = Post.objects.get(id=self.kwargs["pk"])
        return post.author == self.request.user


class Index(TemplateView):
    template_name = "myapp/index.html"

    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(**kwargs)
        post_list = Post.objects.all().order_by("-created_at")
        context = {
            "post_list": post_list,
        }
        return context


class PostCreate(LoginRequiredMixin, CreateView):
    model = Post
    form_class = PostForm
    success_url = reverse_lazy("myapp:index")

    def form_valid(self, form):
        form.instance.author_id = self.request.user.id
        post = form.save(commit=False)
        if post.category_id == 1:
            photo = Photo(image=form.cleaned_data['thumbnail'])
            predicted, percentage = photo.predict()
            post.category_id = predicted

        return super(PostCreate, self).form_valid(form)

    def get_success_url(self):
        messages.success(self.request, "Postを登録しました。")
        return resolve_url("myapp:index")


class PostDetail(DetailView):
    model = Post

    def get_context_data(self, **kwargs):
        detail_data = Post.objects.get(id=self.kwargs["pk"])
        category_posts = Post.objects.filter(category=detail_data.category).order_by(
            "-created_at"
        )[:5]
        params = {
            "object": detail_data,
            "category_posts": category_posts,
        }
        return params


class PostUpdate(OnlyMyPostMixin, UpdateView):
    model = Post
    form_class = PostForm

    def form_valid(self, form):
        post = form.save(commit=False)
        if post.category_id == 1:
            photo = Photo(image=form.cleaned_data['thumbnail'])
            predicted, percentage = photo.predict()
            post.category_id = predicted

        return super(PostUpdate, self).form_valid(form)

    def get_success_url(self):
        messages.info(self.request, "Postを更新しました。")
        return resolve_url("myapp:post_detail", pk=self.kwargs["pk"])
    

class PostDelete(OnlyMyPostMixin, DeleteView):
    model = Post

    def get_success_url(self):
        messages.info(self.request, "Postを削除しました。")
        return resolve_url("myapp:index")


class PostList(ListView):
    model = Post
    paginate_by = 5

    def get_queryset(self):
        return Post.objects.all().order_by("-created_at")


class Login(LoginView):
    form_class = LoginForm
    template_name = "myapp/login.html"


class Logout(LogoutView):
    template_name = "myapp/logout.html"


class SignUp(CreateView):
    form_class = SignUpForm
    template_name = "myapp/signup.html"
    success_url = reverse_lazy("myapp:index")

    def form_valid(self, form):
        user = form.save()

        login(self.request, user, backend='django.contrib.auth.backends.ModelBackend')

        self.object = user
        messages.info(self.request, "ユーザー登録をしました。")
        return HttpResponseRedirect(self.get_success_url())


@login_required
def Like_add(request, post_id):
    post = Post.objects.get(id=post_id)
    is_liked = Like.objects.filter(user=request.user, post=post_id).count()
    if is_liked > 0:
        messages.info(request, "すでにお気に入りに追加済みです。")
        return redirect("myapp:post_detail", post.id)
    like = Like()
    like.user = request.user
    like.post = post
    like.save()

    messages.success(request, "お気に入りに追加しました！")
    return redirect("myapp:post_detail", post.id)


class CategoryList(ListView):
    model = Category


class CategoryDetail(DetailView):
    model = Category
    slug_field = "name_en"
    slug_url_kwarg = "name_en"

    def get_context_data(self, *args, **kwargs):
        detail_data = Category.objects.get(name_en=self.kwargs["name_en"])
        category_posts = Post.objects.filter(category=detail_data.id).order_by(
            "-created_at"
        )

        params = {
            "object": detail_data,
            "category_posts": category_posts,
        }

        return params


def Search(request):
    if request.method == "POST":
        searchform = SearchForm(request.POST)

        if searchform.is_valid():
            freeword = searchform.cleaned_data["freeword"]
            search_list = Post.objects.filter(
                Q(title__icontains=freeword) | Q(content__icontains=freeword)
            )

        params = {
            "search_list": search_list,
        }

        return render(request, "myapp/search.html", params)


def predict(request):

    form = PhotoForm(request.POST, request.FILES)

    photo = Photo(image=form.cleaned_data['image'])
    predicted, percentage = photo.predict()

    context = {
        'photo_name': photo.image.name,
        'photo_data': photo.image_src(),
        'predicted': predicted,
        'percentage': percentage,
    }

    return HttpResponse(template.render(context, request))


def guest_login(request):
    guest_user = CustomUser.objects.get(email='guest@guest.guest')
    login(request, guest_user, backend='django.contrib.auth.backends.ModelBackend')
    
    return redirect('myapp:index')
    